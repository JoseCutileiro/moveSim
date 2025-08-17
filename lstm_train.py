# lstm_train.py
import re
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# --------------------
# Utilidades de dados
# --------------------
def carregar_trajetorias(path_txt: str):
    trajs = []
    with open(path_txt, "r", encoding="utf-8") as f:
        for linha in f:
            if linha.startswith("Traj:"):
                coords = linha.strip().split(":")[1].strip()
                pares = re.findall(r"\((-?\d+),\s*(-?\d+)\)", coords)
                traj = [(int(x), int(y)) for x, y in pares]
                if len(traj) >= 1:
                    trajs.append(np.array(traj, dtype=np.float32))
    return trajs

def build_sequences(trajs, seq_len, pred_len, mean, std):
    """
    Cria amostras mesmo quando NÃO há todos os 'pred_len' futuros.
    X: [N, seq_len, 2]        (normalizado; left-pad com o primeiro valor do slice)
    Y: [N, pred_len, 2]       (normalizado; zeros onde não há futuro)
    M: [N, pred_len]          (máscara 1 onde há target válido, 0 caso contrário)
    """
    X, Y, M = [], [], []
    for t in trajs:
        t_norm = (t - mean) / std
        L = len(t_norm)
        if L == 0:
            continue

        # Escolhemos janelas começando em start_idx; a entrada é t_norm[start_idx : start_idx+seq_len]
        # Se faltar passado, fazemos left-pad; se faltar futuro, mascaramos.
        max_start = max(0, L - 1)  # podemos começar até ao penúltimo índice
        for start_idx in range(0, max_start + 1):
            # ----- X (passado) -----
            x_slice = t_norm[start_idx : start_idx + seq_len]
            if len(x_slice) < seq_len:
                # left-pad com o primeiro valor do slice (repetição)
                first = x_slice[0:1] if len(x_slice) > 0 else (t_norm[0:1] if L > 0 else np.zeros((1,2), np.float32))
                pad = np.repeat(first, seq_len - len(x_slice), axis=0)
                x_full = np.concatenate([pad, x_slice], axis=0)
            else:
                x_full = x_slice

            # ----- Y (futuro) + M (máscara) -----
            fut_start = start_idx + seq_len
            avail = max(0, L - fut_start)  # quantos futuros existem de facto
            y_full = np.zeros((pred_len, 2), dtype=np.float32)
            m_full = np.zeros((pred_len,), dtype=np.float32)
            if avail > 0:
                take = min(pred_len, avail)
                y_full[:take] = t_norm[fut_start : fut_start + take]
                m_full[:take] = 1.0
            else:
                # sem futuros → esta amostra não ajuda; salta
                continue

            X.append(x_full)
            Y.append(y_full)
            M.append(m_full)

    if len(X) == 0:
        return np.empty((0, seq_len, 2), np.float32), np.empty((0, pred_len, 2), np.float32), np.empty((0, pred_len), np.float32)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    M = np.array(M, dtype=np.float32)
    return X, Y, M

# --------------------
# Pesos da loss por horizonte
# --------------------
def make_horizon_weights(H, scheme="exp_mix", gamma=1.0, beta=0.7):
    """
    w[k] para k=0..H-1 (mais peso no curto prazo, sem zerar o longo).
    exp_mix: w = beta*exp(-gamma*k/(H-1)) + (1-beta)
    Normalizado para somar 1.
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

# --------------------
# Modelo LSTM (1-step)
# --------------------
class TrajLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)      # [B, T, H]
        last = out[:, -1, :]       # [B, H]
        pred = self.fc(last)       # [B, 2]
        return pred

# --------------------
# Treino multi-passo com rollout + máscara
# --------------------
def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trajs = carregar_trajetorias(args.data)
    if not trajs:
        raise RuntimeError("Nenhuma trajetória válida foi encontrada.")

    # Normalização global
    all_pts = np.concatenate(trajs, axis=0)
    mean = all_pts.mean(axis=0)
    std = all_pts.std(axis=0) + 1e-8

    X, Y, M = build_sequences(trajs, args.seq_len, args.pred_len, mean, std)
    if len(X) < 1:
        raise RuntimeError("Ainda não há amostras suficientes mesmo com máscara. Reduza --seq_len.")

    # Shuffle e split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, Y, M = X[idx], Y[idx], M[idx]

    n_train = int(len(X) * 0.8)
    X_train, Y_train, M_train = X[:n_train], Y[:n_train], M[:n_train]
    X_val,   Y_val,   M_val   = X[n_train:], Y[n_train:], M[n_train:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajLSTM(input_size=2, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = make_horizon_weights(args.pred_len, scheme=args.weight_scheme,
                                   gamma=args.weight_gamma, beta=args.weight_beta).to(device)

    def batches(*arrays, bs):
        n = len(arrays[0])
        for i in range(0, n, bs):
            yield [a[i:i+bs] for a in arrays]

    best_val = float("inf")
    best_state = None
    patience = args.patience
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # ----------------- Treino -----------------
        model.train()
        tr_loss_sum, tr_count = 0.0, 0

        for xb, yb, mb in batches(X_train, Y_train, M_train, bs=args.batch_size):
            xb_t = torch.from_numpy(xb).to(device)      # [B,T,2]
            yb_t = torch.from_numpy(yb).to(device)      # [B,H,2]
            mb_t = torch.from_numpy(mb).to(device)      # [B,H]
            opt.zero_grad()

            seq = xb_t.clone()
            total_loss = 0.0

            for h in range(args.pred_len):
                pred_h = model(seq)                      # [B,2]
                # MSE por amostra (média nas coords)
                mse_i = ((pred_h - yb_t[:, h, :]) ** 2).mean(dim=1)  # [B]
                mask_h = mb_t[:, h]                      # [B]
                valid = mask_h.sum()
                if valid.item() > 0:
                    # média apenas nos válidos (evita enviesar batches com poucos futuros)
                    loss_h = (mse_i * mask_h).sum() / (valid + 1e-8)
                    total_loss = total_loss + weights[h] * loss_h
                # atualizar janela com a previsão (detach quebra o grad no input futuro)
                seq = torch.cat([seq[:, 1:, :], pred_h.detach().unsqueeze(1)], dim=1)

            total_loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            tr_loss_sum += total_loss.item() * len(xb)
            tr_count += len(xb)

        tr_loss = tr_loss_sum / max(tr_count, 1)

        # ----------------- Validação -----------------
        model.eval()
        with torch.no_grad():
            val_loss_sum, val_count = 0.0, 0
            for xb, yb, mb in batches(X_val, Y_val, M_val, bs=args.batch_size):
                xb_t = torch.from_numpy(xb).to(device)
                yb_t = torch.from_numpy(yb).to(device)
                mb_t = torch.from_numpy(mb).to(device)

                seq = xb_t.clone()
                loss_seq = 0.0
                for h in range(args.pred_len):
                    pred_h = model(seq)
                    mse_i = ((pred_h - yb_t[:, h, :]) ** 2).mean(dim=1)
                    mask_h = mb_t[:, h]
                    valid = mask_h.sum()
                    if valid.item() > 0:
                        loss_h = (mse_i * mask_h).sum() / (valid + 1e-8)
                        loss_seq = loss_seq + weights[h] * loss_h
                    seq = torch.cat([seq[:, 1:, :], pred_h.unsqueeze(1)], dim=1)

                val_loss_sum += loss_seq.item() * len(xb)
                val_count += len(xb)

            vb_loss = val_loss_sum / max(val_count, 1)

        print(f"Epoch {epoch:03d} | train {tr_loss:.6f} | val {vb_loss:.6f}")

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
                "normalization": {"mean": mean.tolist(), "std": std.tolist()},
                "val_loss": float(best_val),
                "weights": weights.detach().cpu().tolist(),
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping.")
            break

    # Guarda
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.out)
    print(f"Modelo guardado em: {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="trajetoriasClean.txt")
    parser.add_argument("--out", type=str, default="lstm_model.pt")
    parser.add_argument("--seq_len", type=int, default=10)     # passado
    parser.add_argument("--pred_len", type=int, default=50)    # horizonte otimizado
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_scheme", type=str, default="exp_mix",
                        choices=["exp_mix", "linear_mix", "uniform"])
    parser.add_argument("--weight_gamma", type=float, default=1.0)
    parser.add_argument("--weight_beta", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
