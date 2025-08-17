# lstm_train_deltas.py
# Treino LSTM para prever deltas (Δx, Δy) de trajetória com rollout + máscara + scheduled sampling.
import re
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --------------------
# Utilidades de dados
# --------------------
def carregar_trajetorias(path_txt: str):
    """
    Lê um .txt com linhas do tipo:
      Traj: (x1,y1),(x2,y2),...
    Retorna: lista de arrays [L,2] em float32.
    """
    trajs = []
    with open(path_txt, "r", encoding="utf-8") as f:
        for linha in f:
            if linha.startswith("Traj:"):
                coords = linha.strip().split(":")[1].strip()
                pares = re.findall(r"\((-?\d+),\s*(-?\d+)\)", coords)
                traj = [(int(x), int(y)) for x, y in pares]
                if len(traj) >= 2:  # precisa de pelo menos dois pontos para ter deltas
                    trajs.append(np.array(traj, dtype=np.float32))
    return trajs


def traj_to_deltas(traj_xy: np.ndarray) -> np.ndarray:
    """
    Converte posições absolutas [L,2] em deltas [L,2], com d[0]=(0,0).
    d[t] = xy[t] - xy[t-1]
    """
    d = np.zeros_like(traj_xy)
    d[1:] = traj_xy[1:] - traj_xy[:-1]
    return d


def make_horizon_weights(H, scheme="exp_mix", gamma=1.0, beta=0.7):
    """
    Pesos w[k] para k=0..H-1 (mais peso no curto prazo, sem zerar o longo).
    exp_mix: w = beta*exp(-gamma*k/(H-1)) + (1-beta). Normaliza para somar 1.
    """
    k = np.arange(H, dtype=np.float32)
    if scheme == "exp_mix":
        base = np.exp(-gamma * k / max(H - 1, 1))
        w = beta * base + (1.0 - beta)
    elif scheme == "linear_mix":
        base = 1.0 - k / max(H - 1, 1)
        w = beta * base + (1.0 - beta)
    else:  # uniforme
        w = np.ones(H, dtype=np.float32)
    w = w / w.sum()
    return torch.tensor(w, dtype=torch.float32)


def build_sequences_from_deltas(
    trajs_xy,
    seq_len: int,
    pred_len: int,
    mean_delta: np.ndarray,
    std_delta: np.ndarray,
):
    """
    Cria amostras usando **DELTAS** (não posições).

    Para cada trajectória:
      - D = deltas [L,2], D[0]=(0,0)
      - Para janela que começa em start:
          X = D[start : start+seq_len]                  (left-pad com D[start] se faltar)
          Y = D[start+seq_len : start+seq_len+pred_len] (zeros onde não há futuro)
          M = máscara 1 onde Y existe

      - anchor = último ponto ABSOLUTO do passado (para reconstruir posições nas métricas)

    Retorna:
      X [N, seq_len, 2]  (normalizado)
      Y [N, pred_len, 2] (normalizado)
      M [N, pred_len]
      anchors [N, 2]     (absoluto, não normalizado)
    """
    X, Y, M, A = [], [], [], []
    for xy in trajs_xy:
        L = len(xy)
        if L < 2:
            continue
        D = traj_to_deltas(xy)  # [L,2] (bruto)

        max_start = max(0, L - 1)  # até penúltimo índice é válido começar
        for start in range(0, max_start + 1):
            # ----- X (passado) -----
            x_slice = D[start : start + seq_len]
            if len(x_slice) < seq_len:
                first = x_slice[0:1] if len(x_slice) > 0 else D[start:start+1]
                pad = np.repeat(first, seq_len - len(x_slice), axis=0)
                x_full = np.concatenate([pad, x_slice], axis=0)
            else:
                x_full = x_slice

            # ----- Y (futuro) + M (máscara) -----
            fut_start = start + seq_len
            avail = max(0, L - fut_start)
            y_full = np.zeros((pred_len, 2), dtype=np.float32)
            m_full = np.zeros((pred_len,), dtype=np.float32)
            if avail > 0:
                take = min(pred_len, avail)
                y_full[:take] = D[fut_start : fut_start + take]
                m_full[:take] = 1.0
            else:
                continue  # sem futuro → ignora

            # Âncora: último ponto absoluto do passado
            # Índice do último ponto do passado = min(fut_start-1, L-1)
            last_past_idx = min(fut_start - 1, L - 1)
            anchor = xy[last_past_idx]

            # Normaliza **DELTAS**
            x_norm = (x_full - mean_delta) / std_delta
            y_norm = (y_full - mean_delta) / std_delta

            X.append(x_norm)
            Y.append(y_norm)
            M.append(m_full)
            A.append(anchor)

    if not X:
        return (
            np.empty((0, seq_len, 2), np.float32),
            np.empty((0, pred_len, 2), np.float32),
            np.empty((0, pred_len), np.float32),
            np.empty((0, 2), np.float32),
        )

    return (
        np.asarray(X, np.float32),
        np.asarray(Y, np.float32),
        np.asarray(M, np.float32),
        np.asarray(A, np.float32),
    )


# --------------------
# Modelo LSTM (1-step)
# --------------------
class TrajLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)         # [B, T, H]
        last = out[:, -1, :]          # [B, H]
        pred_delta = self.fc(last)    # [B, 2]  (delta no espaço normalizado)
        return pred_delta


# --------------------
# Métricas (ADE / FDE)
# --------------------
def ade_fde_from_deltas(
    pred_deltas_norm, tgt_deltas_norm, mask, anchors, mean_delta, std_delta
):
    """
    Calcula ADE/FDE em coordenadas absolutas (não normalizadas).
    pred_deltas_norm, tgt_deltas_norm: [N, H, 2]
    mask: [N, H]
    anchors: [N, 2] (pos absoluto no fim do passado)
    """
    # Desnormaliza deltas
    pred = pred_deltas_norm * std_delta + mean_delta
    tgt = tgt_deltas_norm * std_delta + mean_delta

    # Reconstrói posições cumulando deltas a partir da âncora
    pred_pos = np.cumsum(pred, axis=1) + anchors[:, None, :]
    tgt_pos = np.cumsum(tgt, axis=1) + anchors[:, None, :]

    # Erros euclidianos por passo
    err = np.linalg.norm(pred_pos - tgt_pos, axis=2)  # [N,H]

    # Aplica máscara
    valid = mask > 0.0
    if not np.any(valid):
        return np.nan, np.nan

    # ADE: média em todos os passos válidos
    ade = err[valid].mean()

    # FDE: último passo válido por amostra
    fde_vals = []
    for i in range(err.shape[0]):
        v = np.where(valid[i])[0]
        if len(v) > 0:
            fde_vals.append(err[i, v[-1]])
    fde = float(np.mean(fde_vals)) if fde_vals else np.nan
    return float(ade), float(fde)


# --------------------
# Treino
# --------------------
def train(args):
    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Carrega trajetórias
    trajs_xy = carregar_trajetorias(args.data)
    if not trajs_xy:
        raise RuntimeError("Nenhuma trajetória válida foi encontrada.")

    # ---- Split por TRAJETÓRIA (sem leakage) ----
    ids = np.arange(len(trajs_xy))
    np.random.shuffle(ids)
    n_train = int(0.8 * len(ids))
    tr_ids, va_ids = ids[:n_train], ids[n_train:]

    trajs_tr = [trajs_xy[i] for i in tr_ids]
    trajs_va = [trajs_xy[i] for i in va_ids]

    # ---- Normalização calculada SÓ no TREINO (em DELTAS) ----
    deltas_tr = np.concatenate([traj_to_deltas(xy) for xy in trajs_tr], axis=0)
    mean_delta = deltas_tr.mean(axis=0)
    std_delta = deltas_tr.std(axis=0) + 1e-8

    # ---- Constrói janelas (DELTAS normalizados) ----
    X_tr, Y_tr, M_tr, A_tr = build_sequences_from_deltas(
        trajs_tr, args.seq_len, args.pred_len, mean_delta, std_delta
    )
    X_va, Y_va, M_va, A_va = build_sequences_from_deltas(
        trajs_va, args.seq_len, args.pred_len, mean_delta, std_delta
    )

    if len(X_tr) == 0:
        raise RuntimeError("Sem amostras após o split. Reduza --seq_len ou verifique os dados.")

    print(f"Train samples: {len(X_tr)} | Val samples: {len(X_va)}")

    # ---- Modelo/otimizador ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajLSTM(input_size=2, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = make_horizon_weights(
        args.pred_len, scheme=args.weight_scheme, gamma=args.weight_gamma, beta=args.weight_beta
    ).to(device)

    # ---- Helpers ----
    def batches(*arrays, bs):
        n = len(arrays[0])
        idx = np.arange(n)
        np.random.shuffle(idx)
        for i in range(0, n, bs):
            sl = idx[i : i + bs]
            yield [a[sl] for a in arrays]

    def as_torch(*arrays):
        return [torch.from_numpy(a).to(device) for a in arrays]

    best_val = float("inf")
    best_state = None
    no_improve = 0

    # Scheduled sampling: prob de usar alvo como próximo input
    def teacher_prob(epoch):
        if args.ss_epochs <= 0:
            return 0.0
        # decai linearmente de ss_p_start → ss_p_end ao longo de ss_epochs
        e = min(epoch - 1, args.ss_epochs)
        return args.ss_p_start + (args.ss_p_end - args.ss_p_start) * (e / max(args.ss_epochs, 1))

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss_sum, tr_cnt = 0.0, 0
        p_tf = teacher_prob(epoch)

        for xb, yb, mb, ab in batches(X_tr, Y_tr, M_tr, A_tr, bs=args.batch_size):
            xb_t, yb_t, mb_t = as_torch(xb, yb, mb)
            opt.zero_grad()

            # seq = janela corrente de DELTAS normalizados
            seq = xb_t.clone()
            total_loss = 0.0

            for h in range(args.pred_len):
                pred_h = model(seq)  # [B,2] delta norm
                # MSE por amostra (média nas coords)
                mse_i = ((pred_h - yb_t[:, h, :]) ** 2).mean(dim=1)  # [B]
                mask_h = mb_t[:, h]  # [B]
                valid = mask_h.sum()
                if valid.item() > 0:
                    loss_h = (mse_i * mask_h).sum() / (valid + 1e-8)
                    total_loss = total_loss + weights[h] * loss_h

                # scheduled sampling: decide qual delta empurrar
                if np.random.rand() < p_tf:
                    next_delta = yb_t[:, h, :]  # usa o alvo
                else:
                    next_delta = pred_h.detach()  # usa a predição (sem grad no input)

                seq = torch.cat([seq[:, 1:, :], next_delta.unsqueeze(1)], dim=1)

            total_loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            tr_loss_sum += total_loss.item() * len(xb)
            tr_cnt += len(xb)

        tr_loss = tr_loss_sum / max(tr_cnt, 1)

        # ----- Validação -----
        model.eval()
        with torch.no_grad():
            val_loss_sum, val_cnt = 0.0, 0
            # para ADE/FDE
            preds_all, tgts_all, masks_all, anchors_all = [], [], [], []

            for xb, yb, mb, ab in batches(X_va, Y_va, M_va, A_va, bs=args.batch_size):
                xb_t, yb_t, mb_t = as_torch(xb, yb, mb)
                seq = xb_t.clone()
                loss_seq = 0.0
                preds_steps = []

                for h in range(args.pred_len):
                    pred_h = model(seq)
                    preds_steps.append(pred_h.cpu().numpy())

                    mse_i = ((pred_h - yb_t[:, h, :]) ** 2).mean(dim=1)
                    mask_h = mb_t[:, h]
                    valid = mask_h.sum()
                    if valid.item() > 0:
                        loss_h = (mse_i * mask_h).sum() / (valid + 1e-8)
                        loss_seq = loss_seq + weights[h] * loss_h

                    seq = torch.cat([seq[:, 1:, :], pred_h.unsqueeze(1)], dim=1)

                val_loss_sum += loss_seq.item() * len(xb)
                val_cnt += len(xb)

                # guarda para métricas
                preds = np.stack(preds_steps, axis=1)  # [B,H,2] norm
                preds_all.append(preds)
                tgts_all.append(yb)
                masks_all.append(mb)
                anchors_all.append(ab)

            vb_loss = val_loss_sum / max(val_cnt, 1)

            # ADE/FDE (em coordenadas reais)
            if preds_all:
                preds_all = np.concatenate(preds_all, axis=0)
                tgts_all = np.concatenate(tgts_all, axis=0)
                masks_all = np.concatenate(masks_all, axis=0)
                anchors_all = np.concatenate(anchors_all, axis=0)
                ade, fde = ade_fde_from_deltas(
                    preds_all, tgts_all, masks_all, anchors_all, mean_delta, std_delta
                )
            else:
                ade, fde = np.nan, np.nan

        print(
            f"Epoch {epoch:03d} | train {tr_loss:.6f} | val {vb_loss:.6f} | ADE {ade:.3f} | FDE {fde:.3f} | p_tf {p_tf:.2f}"
        )

        # Early stopping
        if vb_loss < best_val - 1e-6:
            best_val = vb_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "input_size": 2,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "seq_len": args.seq_len,
                    "pred_len": args.pred_len,
                },
                "normalization": {
                    "mean_delta": mean_delta.tolist(),
                    "std_delta": std_delta.tolist(),
                },
                "val_loss": float(best_val),
                "weights": weights.detach().cpu().tolist(),
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print("Early stopping.")
            break

    # Guarda
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.out)
    print(f"Modelo guardado em: {args.out}")


# --------------------
# Main / CLI
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="trajetoriasClean.txt")
    parser.add_argument("--out", type=str, default="lstm_model_deltas.pt")

    parser.add_argument("--seq_len", type=int, default=10)     # janelas de passado (em deltas)
    parser.add_argument("--pred_len", type=int, default=50)    # horizonte futuro (em deltas)

    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--weight_scheme", type=str, default="exp_mix",
                        choices=["exp_mix", "linear_mix", "uniform"])
    parser.add_argument("--weight_gamma", type=float, default=1.0)
    parser.add_argument("--weight_beta", type=float, default=0.7)

    # Scheduled sampling (teacher forcing)
    parser.add_argument("--ss_epochs", type=int, default=20, help="épocas para decair p_tf")
    parser.add_argument("--ss_p_start", type=float, default=1.0, help="prob inicial de usar alvo")
    parser.add_argument("--ss_p_end", type=float, default=0.0, help="prob final de usar alvo")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
