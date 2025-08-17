# lstm_use.py
import warnings
import torch
import torch.nn as nn
import numpy as np

_GLOBAL = {"model": None, "cfg": None, "mean_delta": None, "std_delta": None,
           "device": "cpu", "loaded_path": None}

class TrajLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, hc=None):
        out, hc = self.lstm(x, hc)      # [B,T,H]
        last = out[:, -1, :]
        pred = self.fc(last)            # delta normalizado
        return pred, hc

def _load(model_path: str, prefer_weights_only: bool = False):
    # Recarrega se o caminho mudou ou se ainda não há modelo
    if _GLOBAL["model"] is not None and _GLOBAL["loaded_path"] == model_path:
        return

    # Nota: o teu checkpoint é um dicionário com 'model_state_dict' e 'normalization'.
    # Se passares weights_only=True aqui, em versões futuras do PyTorch pode falhar.
    # Mantemos o default (False) e suprimimos o aviso para checkpoints confiáveis.
    if prefer_weights_only:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)  # só use se salvar só state_dict
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            ckpt = torch.load(model_path, map_location="cpu")

    if "model_state_dict" in ckpt and "config" in ckpt:
        cfg = ckpt["config"]
        model = TrajLSTM(
            input_size=cfg.get("input_size", 2),
            hidden_size=cfg.get("hidden_size", 64),
            num_layers=cfg.get("num_layers", 2),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        norm = ckpt.get("normalization", {})
        mean_delta = np.array(norm["mean_delta"], dtype=np.float32)
        std_delta = np.array(norm["std_delta"], dtype=np.float32)
    else:
        # Caso seu checkpoint seja só state_dict de um modelo igual
        raise ValueError("Checkpoint inesperado: esperava chaves 'model_state_dict' e 'config'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    _GLOBAL.update({
        "model": model, "cfg": cfg,
        "mean_delta": mean_delta, "std_delta": std_delta,
        "device": device, "loaded_path": model_path
    })

def _as_deltas(history_xy):
    """Converte histórico absoluto em deltas; se <2 pontos, devolve deltas zeros."""
    arr = np.asarray(history_xy, dtype=np.float32)
    if arr.size == 0:
        # sem pontos: retorna um único delta zero e um ponto âncora (0,0)
        return np.zeros((1, 2), np.float32), np.array([0.0, 0.0], np.float32)
    if len(arr) == 1:
        # 1 ponto: deltas = [[0,0]], âncora = ponto único
        return np.zeros((1, 2), np.float32), arr[-1]
    deltas = np.zeros_like(arr)
    deltas[1:] = arr[1:] - arr[:-1]
    return deltas, arr[-1]  # devolve também último absoluto como âncora

def _prep_seq(history_xy, seq_len, mean_delta, std_delta):
    """
    history_xy: lista de (x,y) absolutos
    Retorna:
      seq [1, seq_len, 2] (deltas normalizados)
      last_abs [2] (último ponto absoluto)
    """
    deltas, last_abs = _as_deltas(history_xy)

    # left-pad para seq_len
    if len(deltas) < seq_len:
        pad = np.repeat(deltas[0:1], seq_len - len(deltas), axis=0)
        deltas = np.concatenate([pad, deltas], axis=0)
    else:
        deltas = deltas[-seq_len:]

    deltas_norm = (deltas - mean_delta) / std_delta
    t = torch.from_numpy(deltas_norm).unsqueeze(0).float()  # [1,T,2]
    return t, np.asarray(last_abs, dtype=np.float32)

@torch.no_grad()
def lstm_predict(history, steps=50, model_path="lstm_model_deltas.pt"):
    """
    history: lista de (x,y) absolutos
    steps: nº de passos a prever
    Retorna lista de (x,y) absolutos previstos
    """
    _load(model_path, prefer_weights_only=False)  # mantém False para o teu checkpoint
    model = _GLOBAL["model"]
    cfg = _GLOBAL["cfg"]
    mean_delta = _GLOBAL["mean_delta"]
    std_delta = _GLOBAL["std_delta"]
    device = _GLOBAL["device"]

    seq_len = cfg["seq_len"]
    seq, last_abs = _prep_seq(history, seq_len, mean_delta, std_delta)
    seq = seq.to(device)

    preds = []
    hc = None
    for _ in range(steps):
        pred_norm, hc = model(seq, hc)                      # delta normalizado
        delta = (pred_norm.squeeze(0).cpu().numpy() * std_delta) + mean_delta  # delta real
        next_abs = last_abs + delta
        preds.append((float(next_abs[0]), float(next_abs[1])))

        # prepara próximo passo
        new_norm = (delta - mean_delta) / std_delta
        seq_np = seq.squeeze(0).cpu().numpy()
        seq_np = np.concatenate([seq_np[1:], new_norm.reshape(1, 2)], axis=0)
        seq = torch.from_numpy(seq_np).unsqueeze(0).float().to(device)
        last_abs = next_abs

    return preds

if __name__ == "__main__":
    # Teste rápido
    hist = [(688, 557)]  # até com 1 ponto agora funciona
    print(lstm_predict(hist, steps=5, model_path="lstm_model_deltas.pt"))
