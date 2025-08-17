# lstm_use.py
import torch
import torch.nn as nn
import numpy as np

# Mantém cache global para evitar recarregar a cada chamada
_GLOBAL = {"model": None, "cfg": None, "mean": None, "std": None, "device": "cpu"}

class TrajLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, hc=None):
        out, hc = self.lstm(x, hc)      # out: [B, T, H]
        last = out[:, -1, :]
        pred = self.fc(last)
        return pred, hc

def _load(model_path: str):
    if _GLOBAL["model"] is not None:
        return
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ckpt["config"]
    model = TrajLSTM(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
    ).to("cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean = np.array(ckpt["normalization"]["mean"], dtype=np.float32)
    std = np.array(ckpt["normalization"]["std"], dtype=np.float32)

    _GLOBAL.update({"model": model, "cfg": cfg, "mean": mean, "std": std, "device": "cpu"})

def _prep_seq(history_xy, seq_len, mean, std):
    """
    history_xy: lista de tuplos (x,y) em coordenadas originais.
    Retorna tensor [1, seq_len, 2] normalizado.
    Se o histórico for curto, faz left-pad repetindo o primeiro ponto.
    """
    arr = np.array(history_xy, dtype=np.float32)
    if len(arr) == 0:
        raise ValueError("history vazio.")
    if len(arr) < seq_len:
        pad = np.repeat(arr[0:1], seq_len - len(arr), axis=0)
        arr = np.concatenate([pad, arr], axis=0)
    else:
        arr = arr[-seq_len:]  # últimos seq_len pontos

    arr_norm = (arr - mean) / std
    t = torch.from_numpy(arr_norm).unsqueeze(0)  # [1, T, 2]
    return t

@torch.no_grad()
def lstm_predict(history, steps=50, model_path="lstm_model.pt"):
    """
    history: lista de (x, y) com os pontos percorridos (p.ex. carro.history).
    steps: número de passos a prever.
    Retorna lista de tuplos (x,y) previstos.
    """
    _load(model_path)
    model = _GLOBAL["model"]
    cfg = _GLOBAL["cfg"]
    mean = _GLOBAL["mean"]
    std = _GLOBAL["std"]

    seq_len = cfg["seq_len"]

    # sequência normalizada inicial
    seq = _prep_seq(history, seq_len, mean, std).float()  # [1, T, 2]

    preds = []
    hc = None  # pode-se manter/calc ao longo dos passos
    for _ in range(steps):
        # Executa LSTM e obtém próximo ponto normalizado
        pred_norm, hc = model(seq, hc)
        # Denormaliza
        pred_xy = pred_norm.squeeze(0).cpu().numpy() * std + mean
        x, y = float(pred_xy[0]), float(pred_xy[1])
        preds.append((x, y))
        # Atualiza a sequência: remove o primeiro, acrescenta o novo (em normalizado)
        new_norm = (pred_xy - mean) / std
        seq_np = seq.squeeze(0).cpu().numpy()           # [T,2]
        seq_np = np.concatenate([seq_np[1:], new_norm.reshape(1, 2)], axis=0)
        seq = torch.from_numpy(seq_np).unsqueeze(0).float()

    return preds

# Pequeno teste manual
if __name__ == "__main__":
    # histórico fictício
    hist = [(688, 557), (679, 547), (670, 537), (651, 519), (632, 500)]
    preds = lstm_predict(hist, steps=5, model_path="lstm_model.pt")
    print(preds)
