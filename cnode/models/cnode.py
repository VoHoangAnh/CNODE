"""CNODE: Conditional Neural ODE for longitudinal PD progression forecasting.

Maps paper equations to code:
    Eq. (1)  AlignmentHead            tau_i = f_tau(M_i, C_i), gamma_i = f_gamma(M_i, C_i)
    Eq. (2)  CNODE.align_times        t~_{i,k} = tau_i + gamma_i (t_{i,k} - t_{i,1})
    Eq. (3-4) losses.metadata_weighted_infonce
    Eq. (5-6) RecognitionEncoder      q(z_{i,0} | X_{i,0:n_i}) = N(mu_i, diag(sigma_i^2))
    Eq. (7)  CNODE.integrate          z_i(t~) = ODESolve(g, z_{i,0}, t~)
    Eq. (8-9) decoder                 X^_{i,k} = D(z_i(t~_{i,k}))
    Eq. (10-12) CNODE.forward         L = L_MSE + lambda_KL L_KL + lambda_C L_contrast
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torchdiffeq import odeint

from .losses import gaussian_kl, masked_mse, metadata_weighted_infonce


@dataclass
class CNODEConfig:
    input_dim: int
    cond_dim: int                 # dim(M) + dim(C)
    latent_dim: int = 16
    hidden_dim: int = 64
    enc_hidden_dim: int = 64
    align_hidden_dim: int = 32
    use_alignment: bool = True     # False -> plain latent Neural ODE baseline
    lambda_kl: float = 1.0
    lambda_c: float = 1.0
    contrast_delta: float = 0.5
    contrast_temperature: float = 0.1
    ode_method: str = "dopri5"
    rtol: float = 1e-4
    atol: float = 1e-5


def mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2) -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class AlignmentHead(nn.Module):
    """Eq. (1): two lightweight regressors producing onset shift and progression speed."""

    def __init__(self, cond_dim: int, hidden: int):
        super().__init__()
        self.f_tau = mlp(cond_dim, hidden, 1)
        self.f_gamma = mlp(cond_dim, hidden, 1)
        # Initialise at the identity alignment: tau = 0, gamma = softplus(b) = 1.
        nn.init.zeros_(self.f_tau[-1].weight)
        nn.init.zeros_(self.f_tau[-1].bias)
        nn.init.zeros_(self.f_gamma[-1].weight)
        nn.init.constant_(self.f_gamma[-1].bias, math.log(math.e - 1.0))

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tau = self.f_tau(cond).squeeze(-1)
        gamma = F.softplus(self.f_gamma(cond)).squeeze(-1) + 1e-3   # strictly positive speed
        return tau, gamma


class RecognitionEncoder(nn.Module):
    """Eqs. (5-6): GRU over the observed visits -> Gaussian posterior of z_{i,0}.

    Each step receives the shape features and the aligned elapsed time, so the
    encoder is aware of the (irregular) spacing between observations.
    """

    def __init__(self, input_dim: int, hidden: int, latent_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim + 1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2 * latent_dim)

    def forward(self, x: torch.Tensor, dt: torch.Tensor, lengths: torch.Tensor):
        inp = torch.cat([x, dt.unsqueeze(-1)], -1)
        packed = pack_padded_sequence(inp, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        mu, log_sigma = self.head(h[-1]).chunk(2, -1)
        return mu, 2.0 * log_sigma                    # (mu, log sigma^2)


class _AlignedODEFunc(nn.Module):
    """Vector field written in elapsed time ``s = t - t_{i,1}``.

    With ``t~ = tau + gamma * s`` the chain rule gives dz/ds = gamma * g(z, t~). Solving in
    ``s`` lets every subject in a batch share one integration grid while still evolving
    on its own aligned timeline (exactly equivalent to Eq. 7).
    """

    def __init__(self, g: nn.Module):
        super().__init__()
        self.g = g
        self.tau = None
        self.gamma = None
        self.nfe = 0

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        t_al = (self.tau + self.gamma * s).unsqueeze(-1)
        return self.gamma.unsqueeze(-1) * self.g(torch.cat([z, t_al], -1))


class CNODE(nn.Module):
    def __init__(self, cfg: CNODEConfig):
        super().__init__()
        self.cfg = cfg
        self.align = AlignmentHead(cfg.cond_dim, cfg.align_hidden_dim)
        self.encoder = RecognitionEncoder(cfg.input_dim, cfg.enc_hidden_dim, cfg.latent_dim)
        g = nn.Sequential(nn.Linear(cfg.latent_dim + 1, cfg.hidden_dim), nn.Tanh(),
                          nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.Tanh(),
                          nn.Linear(cfg.hidden_dim, cfg.latent_dim))
        self.odefunc = _AlignedODEFunc(g)
        self.decoder = mlp(cfg.latent_dim, cfg.hidden_dim, cfg.input_dim)

    # ------------------------------------------------------------------ helpers
    def alignment(self, batch: Dict[str, torch.Tensor]):
        if not self.cfg.use_alignment:
            B = batch["times"].shape[0]
            zeros = batch["times"].new_zeros(B)
            return zeros, zeros + 1.0
        cond = torch.cat([batch["M"], batch["C"]], -1)
        return self.align(cond)

    @staticmethod
    def align_times(times: torch.Tensor, tau: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Eq. (2). ``times`` are already relative to the first visit (t_{i,1} = 0)."""
        elapsed = times - times[:, :1]
        return tau[:, None] + gamma[:, None] * elapsed

    def integrate(self, z0, times, tau, gamma) -> torch.Tensor:
        """Eq. (7) evaluated at every visit time. Returns z of shape (B, K, latent)."""
        elapsed = times - times[:, :1]
        grid, inverse = torch.unique(elapsed, sorted=True, return_inverse=True)
        if grid[0] > 0:  # make sure the solver starts at s = 0 (the first visit)
            grid = torch.cat([grid.new_zeros(1), grid])
            inverse = inverse + 1
        self.odefunc.tau, self.odefunc.gamma = tau, gamma
        if grid.numel() == 1:
            zs = z0.unsqueeze(0)
        else:
            zs = odeint(self.odefunc, z0, grid, method=self.cfg.ode_method,
                        rtol=self.cfg.rtol, atol=self.cfg.atol)    # (T, B, latent)
        B, K = times.shape
        b_idx = torch.arange(B, device=times.device)[:, None].expand(B, K)
        return zs[inverse, b_idx]                                    # (B, K, latent)

    def encode(self, batch, tau, gamma):
        ctx = batch["context"]
        lengths = ctx.sum(1)
        t_al = self.align_times(batch["times"], tau, gamma)
        dt = (t_al - tau[:, None]) * ctx                             # aligned elapsed time
        return self.encoder(batch["X"] * ctx.unsqueeze(-1), dt, lengths)

    # ------------------------------------------------------------------ API
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training step: reconstruct/forecast all visits and return the Eq. (12) objective."""
        tau, gamma = self.alignment(batch)
        mu, logvar = self.encode(batch, tau, gamma)
        z0 = mu + torch.randn_like(mu) * (0.5 * logvar).exp()        # reparameterisation
        z = self.integrate(z0, batch["times"], tau, gamma)
        x_hat = self.decoder(z)

        valid = batch["valid"].float()
        l_mse = masked_mse(batch["X"], x_hat, valid)
        l_kl = gaussian_kl(mu, logvar)
        loss = l_mse + self.cfg.lambda_kl * l_kl
        l_c = x_hat.new_zeros(())
        if self.cfg.use_alignment and self.cfg.lambda_c > 0:
            v = batch["valid"]
            t_al = self.align_times(batch["times"], tau, gamma)
            cond = torch.cat([batch["M"], batch["C"]], -1)
            sid = torch.arange(v.shape[0], device=v.device)[:, None].expand_as(v)
            l_c = metadata_weighted_infonce(
                batch["X"][v], t_al[v], cond[:, None, :].expand(-1, v.shape[1], -1)[v], sid[v],
                delta=self.cfg.contrast_delta, temperature=self.cfg.contrast_temperature)
            loss = loss + self.cfg.lambda_c * l_c
        return {"loss": loss, "mse": l_mse.detach(), "kl": l_kl.detach(),
                "contrast": l_c.detach(), "pred": x_hat}

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Deterministic forecast (posterior mean) at every visit time in the batch."""
        tau, gamma = self.alignment(batch)
        mu, _ = self.encode(batch, tau, gamma)
        return self.decoder(self.integrate(mu, batch["times"], tau, gamma))

    @torch.no_grad()
    def trajectory(self, batch: Dict[str, torch.Tensor], horizon: float, n_points: int = 50):
        """Continuous forecast on a dense grid of elapsed times [0, horizon] (years).

        Returns (elapsed grid (T,), aligned times (B, T), predicted features (B, T, D)).
        """
        tau, gamma = self.alignment(batch)
        mu, _ = self.encode(batch, tau, gamma)
        grid = torch.linspace(0.0, horizon, n_points, device=mu.device)
        times = grid.unsqueeze(0).expand(mu.shape[0], -1).contiguous()
        x = self.decoder(self.integrate(mu, times, tau, gamma))
        return grid, self.align_times(times, tau, gamma), x
