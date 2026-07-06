"""
TPC Shape Predictor
===================
Loads the pre-trained UDE (Universal Differential Equation) model from
the Hybrid-TPC-Model and exposes a clean prediction interface.

The model architecture follows "Group IV – UTPC + constrained residual":
  - ESMTempEncoder_MLP  : patch-based Transformer that maps ESM embeddings
                           into a latent code z ∈ R^64
  - ParamHead           : maps z → (P_max, E), two UTPC parameters
  - ResidualMLP         : learns a data-driven correction to the ODE derivative
  - UTPC ODE            : physics prior dy/dT ~ Eppley-style kinetics

The OGT is now supplied by :class:`mgtp.OGTPredictor` instead of the
original CSV-based noise simulator.

Inputs
------
esm_embedding : ndarray, shape (emb_len,)
    Mean-pooled ESM-2 protein language model embedding for the proteome.
ogt_c         : float
    Optimal growth temperature in °C (from OGT MLP or provided externally).
temperatures  : ndarray
    Target temperature points in °C (ascending, ≥ 2 values).

Outputs
-------
pred_shape : ndarray
    Normalised TPC (peak = 1).  Multiply by an absolute peak rate (from FBA
    or literature) to obtain the actual growth-rate curve.
params     : dict  {"Pmax": float, "ToptC": float, "E": float}
"""

import math
import pickle
import warnings
from pathlib import Path
from typing import Union, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

# ── Architecture constants (must match training) ──────────────────────────────
_ATTN_DIM = 128
_N_HEADS  = 4
_N_LAYERS = 1
_DROPOUT  = 0.1
_Z_DIM    = 64
_P_MAX    = 10.0
_E_MIN    = 3.0
_E_MAX    = 60.0
_X_MIN    = -60.0


# ── Neural-network modules (identical to training code) ───────────────────────

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[: x.size(0)]


class _ESMTempEncoder(nn.Module):
    def __init__(self, emb_len, attn_dim=_ATTN_DIM, n_heads=_N_HEADS,
                 n_layers=_N_LAYERS, out_dim=_Z_DIM, p_drop=_DROPOUT,
                 n_patches=64, mlp_hidden=128, tfeat_dim=2):
        super().__init__()
        self.n_patches = n_patches
        self.patch_dim = emb_len // n_patches
        self.patch_mlp = nn.Sequential(
            nn.Linear(self.patch_dim, mlp_hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(mlp_hidden, attn_dim),
        )
        self.temp_proj = nn.Sequential(
            nn.Linear(tfeat_dim, attn_dim), nn.ReLU(), nn.Dropout(p_drop),
        )
        self.pos = _PositionalEncoding(attn_dim, max_len=n_patches + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=attn_dim, nhead=n_heads, dim_feedforward=attn_dim * 2,
            dropout=p_drop, batch_first=False, norm_first=True, activation="gelu",
        )
        self.tx  = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(attn_dim, out_dim)

    def forward(self, emb_vec, tfeat):
        B, L = emb_vec.shape
        x = self.patch_mlp(
            emb_vec.view(B, self.n_patches, self.patch_dim)
        )
        tok = self.temp_proj(tfeat).unsqueeze(1)
        x   = torch.cat([tok, x], dim=1).transpose(0, 1)
        return self.out(self.tx(self.pos(x))[0])


class _ParamHead(nn.Module):
    def __init__(self, in_dim=_Z_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(_DROPOUT),
            nn.Linear(hidden, 2),
        )

    def forward(self, z):
        return self.net(z)


class _ResidualMLP(nn.Module):
    def __init__(self, z_dim=_Z_DIM, hidden=128, out_scale=1e-3, y_clip=50.0):
        super().__init__()
        self.y_clip = y_clip
        self.net = nn.Sequential(
            nn.Linear(2 + z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        self.out_scale = out_scale

    def forward(self, t_norm, y, z):
        yc = torch.clamp(y, -self.y_clip, self.y_clip)
        x  = torch.cat([t_norm.view(-1, 1), yc.view(-1, 1), z], dim=1)
        return self.net(x).squeeze(1) * self.out_scale


class _ODEFunc(nn.Module):
    def __init__(self, residual: _ResidualMLP, hard_gate: bool = True,
                 detach_z: bool = True):
        super().__init__()
        self.residual  = residual
        self.hard_gate = hard_gate
        self.detach_z  = detach_z
        self.params = self.z = self.t_mean = self.t_std = None

    def set_context(self, Pmax, ToptC, E, z, t_mean, t_std):
        self.params = (Pmax, ToptC, E)
        self.z, self.t_mean, self.t_std = z, t_mean, t_std

    def forward(self, tK, y):
        t_in = tK.view(1) if tK.dim() == 0 else tK
        y_in = y.view(1)  if y.dim()  == 0 else y
        Tc               = t_in - 273.15
        Pmax, ToptC, E   = self.params
        x_val = (Tc - ToptC) / (E + 1e-8)
        x_eff = torch.clamp(x_val, min=_X_MIN, max=1.0)
        dbase = -(Pmax / (E + 1e-8)) * torch.exp(x_eff) * x_eff
        t_norm   = (t_in - self.t_mean) / (self.t_std + 1e-8)
        z_use    = self.z.detach() if self.detach_z else self.z
        dres_raw = self.residual(t_norm, y_in, z_use.expand(t_in.size(0), -1))
        dres     = (torch.where(Tc > ToptC, -F.softplus(dres_raw), dres_raw)
                    if self.hard_gate else dres_raw)
        dy       = dbase + dres
        return torch.where((x_val > 1.0) & (y_in <= 0.0),
                           torch.zeros_like(dy), dy)


class _UDEModel(nn.Module):
    def __init__(self, encoder, head, residual,
                 hard_gate=True, detach_z=True):
        super().__init__()
        self.encoder = encoder
        self.head    = head
        self.odefunc = _ODEFunc(residual, hard_gate, detach_z)

    def forward_curve(self, emb_std, Tk_vec, ogtK, t_mean_k, t_std_k, dev):
        if not torch.is_tensor(emb_std):
            emb_std = torch.tensor(emb_std, dtype=torch.float32, device=dev)
        if not torch.is_tensor(Tk_vec):
            Tk_vec  = torch.tensor(Tk_vec,  dtype=torch.float32, device=dev)
        ogtK = torch.tensor(float(ogtK), dtype=torch.float32, device=dev)

        tfeat = torch.zeros((1, 2), dtype=torch.float32, device=dev)
        z     = self.encoder(emb_std.view(1, -1), tfeat)
        raw   = self.head(z).view(-1)

        Pmax  = _P_MAX * torch.sigmoid(raw[0]) + 1e-6
        E     = _E_MIN + (_E_MAX - _E_MIN) * torch.sigmoid(raw[1])
        ToptC = ogtK - 273.15

        self.odefunc.set_context(Pmax, ToptC, E, z, t_mean_k, t_std_k)

        Tc0 = Tk_vec[:1] - 273.15
        x0  = torch.clamp((Tc0 - ToptC) / (E + 1e-8), min=_X_MIN, max=1.0)
        y0  = torch.clamp(Pmax * torch.exp(x0) * (1.0 - x0), min=0.0).view(-1)

        try:
            from torchdiffeq import odeint as _td_odeint
            traj = _td_odeint(self.odefunc, y0, Tk_vec,
                              rtol=1e-5, atol=1e-6, method="dopri5")
        except Exception:
            # Fallback: 4th-order Runge-Kutta
            y, ys = y0, [y0.clone()]
            for i in range(1, len(Tk_vec)):
                h  = (Tk_vec[i] - Tk_vec[i - 1]).to(y.dtype)
                ti = Tk_vec[i - 1]
                k1 = self.odefunc(ti,         y)
                k2 = self.odefunc(ti + 0.5*h, y + 0.5*h*k1)
                k3 = self.odefunc(ti + 0.5*h, y + 0.5*h*k2)
                k4 = self.odefunc(ti + h,     y + h*k3)
                y  = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                ys.append(y.clone())
            traj = torch.stack(ys, dim=0)

        y_pred = torch.clamp(traj.squeeze(-1), min=0.0)
        return y_pred, (Pmax, ToptC, E)


# ── Public interface ───────────────────────────────────────────────────────────

def load_tpc_model(model_dir: Union[str, Path]):
    """
    Load the pre-trained TPC shape model.

    Parameters
    ----------
    model_dir : path-like
        Directory containing ``checkpoint.pt`` and ``esm_scaler.pkl``.

    Returns
    -------
    predictor : TPCShapePredictor
    """
    return TPCShapePredictor(model_dir)


class TPCShapePredictor:
    """
    Predict normalised TPC shape from ESM embeddings + OGT.

    The OGT is supplied by :class:`mgtp.OGTPredictor` (replacing the original
    ±5 °C noise simulator).  The output is a normalised curve (peak = 1)
    that must be anchored to an absolute rate (e.g. from FBA).

    Parameters
    ----------
    model_dir : path-like
        Directory containing ``checkpoint.pt`` and ``esm_scaler.pkl``.
    """

    def __init__(self, model_dir: Union[str, Path]):
        model_dir = Path(model_dir)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = dev

        ckpt = torch.load(model_dir / "checkpoint.pt", map_location=dev)
        hp   = ckpt.get("hyperparams", {})

        emb_len   = ckpt["emb_len"]
        n_patches = ckpt["n_patches"]
        z_dim     = hp.get("Z_DIM",    _Z_DIM)
        y_clip    = hp.get("Y_CLIP",   50.0)
        hard_gate = hp.get("HARD_NO_INCREASE_AFTER_OGT", True)
        attn_dim  = hp.get("ATTN_DIM", _ATTN_DIM)
        n_heads   = hp.get("N_HEADS",  _N_HEADS)
        n_layers  = hp.get("N_LAYERS", _N_LAYERS)
        dropout   = hp.get("DROPOUT",  _DROPOUT)

        encoder  = _ESMTempEncoder(emb_len=emb_len, attn_dim=attn_dim,
                                   n_heads=n_heads, n_layers=n_layers,
                                   out_dim=z_dim, p_drop=dropout,
                                   n_patches=n_patches)
        head     = _ParamHead(in_dim=z_dim)
        residual = _ResidualMLP(z_dim=z_dim, y_clip=y_clip)

        encoder.load_state_dict(ckpt["encoder"])
        head.load_state_dict(ckpt["head"])
        residual.load_state_dict(ckpt["residual"])

        self._model = _UDEModel(encoder, head, residual,
                                hard_gate=hard_gate).to(dev).eval()
        for p in self._model.parameters():
            p.requires_grad = False

        self._t_mean_k = torch.tensor(ckpt["t_mean_k"], dtype=torch.float32, device=dev)
        self._t_std_k  = torch.tensor(ckpt["t_std_k"],  dtype=torch.float32, device=dev)
        self.esm_cols  = ckpt["esm_cols"]
        self.emb_len   = emb_len

        with open(model_dir / "esm_scaler.pkl", "rb") as fh:
            self._scaler = pickle.load(fh)

    # ------------------------------------------------------------------
    def predict(
        self,
        esm_embedding: np.ndarray,
        ogt_c: float,
        temperatures: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Predict normalised TPC for a single organism.

        Parameters
        ----------
        esm_embedding : ndarray, shape (emb_len,)
            Mean-pooled ESM-2 embedding for the organism's proteome.
        ogt_c : float
            Optimal growth temperature in °C.
        temperatures : ndarray
            Temperature query points in °C (at least 2 values).

        Returns
        -------
        pred_shape : ndarray
            Normalised growth rate at each temperature (peak = 1).
        params : dict
            ``{"Pmax": float, "ToptC": float, "E": float}``
        """
        emb  = np.asarray(esm_embedding, dtype=np.float32).ravel()
        temps = np.asarray(temperatures, dtype=np.float32)
        if len(temps) < 2:
            raise ValueError("temperatures must contain at least 2 points")

        emb_std = self._scaler.transform(emb.reshape(1, -1))[0].astype(np.float32)
        Tk_arr  = np.sort(temps + 273.15).astype(np.float32)
        ogtK    = float(ogt_c) + 273.15

        with torch.no_grad():
            y_raw, (Pmax, ToptC, E) = self._model.forward_curve(
                emb_std, Tk_arr, ogtK,
                self._t_mean_k, self._t_std_k, self._device,
            )

        y_np = y_raw.cpu().numpy().astype(np.float32)
        peak = float(np.max(y_np))
        norm = y_np / (peak if abs(peak) > 1e-8 else 1.0)

        return norm, {
            "Pmax":  float(Pmax.item()),
            "ToptC": float(ToptC.item()),
            "E":     float(E.item()),
        }
