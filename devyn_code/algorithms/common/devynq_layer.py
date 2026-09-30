from typing import Dict, Optional

import numpy as np
import torch as th
from torch import nn


class DevynQLayerBase(nn.Module):
    def __init__(
        self,
        muscle_dims: int,
        group_num: int,
        *,
        shaping_start_steps: Optional[int] = None,
        bias: bool = False,
        bias_init: Optional[np.ndarray] = None,
        init_std: float = 1.0,
    ):
        super().__init__()
        assert muscle_dims > 0
        assert group_num > 0

        self.muscle_dims = int(muscle_dims)
        self.muscle_group_num = int(group_num)

        U0 = th.randn(self.muscle_dims, self.muscle_group_num, dtype=th.float32) * float(init_std)
        U0 = U0 / (U0.abs().sum(dim=1, keepdim=True).clamp(min=1e-6))

        self.register_buffer("U0", U0)
        self.U = nn.Parameter(U0.clone(), requires_grad=True)
        self.register_buffer("U_ema", U0.clone())
        self.register_buffer("U_anchor", U0.clone())

        self.shaping_start_steps = shaping_start_steps
        self._shaping_enabled = self.shaping_start_steps is None

        self.has_bias = bool(bias)
        if self.has_bias:
            if bias_init is None:
                b0_np = np.zeros((self.muscle_dims,), dtype=np.float32)
            else:
                b0_np = np.asarray(bias_init, dtype=np.float32)
                assert b0_np.shape == (self.muscle_dims,)
            b0 = th.as_tensor(b0_np, dtype=th.float32)
            self.register_buffer("b0", b0)
            self.b = nn.Parameter(b0.clone(), requires_grad=True)
            self.register_buffer("b_ema", b0.clone())
            self.register_buffer("b_anchor", b0.clone())

        self.register_buffer("dircov_ema", th.eye(self.muscle_dims, dtype=th.float32))
        self.register_buffer("dircov_last", th.eye(self.muscle_dims, dtype=th.float32))
        self.register_buffer("_tiny", th.tensor(1e-8, dtype=th.float32))

    def maybe_enable_shaping(self, num_timesteps: int) -> None:
        if self.shaping_start_steps is None:
            self._shaping_enabled = True
        elif (not self._shaping_enabled) and num_timesteps >= self.shaping_start_steps:
            self._shaping_enabled = True

    def _current_U(self) -> th.Tensor:
        if not self._shaping_enabled:
            Uuse = self.U0
        else:
            Uuse = self.U_ema.detach() + (self.U - self.U.detach())
        return Uuse.to(dtype=self.U.dtype)

    def _current_b(self) -> th.Tensor:
        if not self.has_bias:
            return th.zeros(self.muscle_dims, dtype=self.U.dtype, device=self.U.device)
        if not self._shaping_enabled:
            return self.b0.to(dtype=self.b.dtype)
        return self.b_ema.detach() + (self.b - self.b.detach())

    def _project(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[-1] == self.muscle_group_num
        U = self._current_U().to(dtype=x.dtype)
        b = self._current_b().to(dtype=x.dtype)
        if x.dim() == 1:
            return U @ x + b
        x2d = x.reshape(-1, self.muscle_group_num)
        y2d = x2d @ U.T + b
        return y2d.reshape(*x.shape[:-1], self.muscle_dims)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._project(x)

    def update_direction_cov(
        self,
        directions: th.Tensor,
        *,
        ema_tau: float,
        center: bool = True,
        normalize_rows: bool = True,
        eps: float = 1e-8,
    ) -> None:
        if directions.ndim != 2 or directions.shape[1] != self.muscle_dims:
            raise ValueError(
                f"directions must have shape [B, {self.muscle_dims}], got {tuple(directions.shape)}"
            )

        with th.no_grad():
            d = directions.detach().to(dtype=self.dircov_ema.dtype, device=self.dircov_ema.device)
            if normalize_rows:
                d = d / d.norm(dim=1, keepdim=True).clamp(min=eps)
            if center:
                d = d - d.mean(dim=0, keepdim=True)
            cov = (d.T @ d) / max(1, d.shape[0])
            cov = 0.5 * (cov + cov.T)
            self.dircov_last.copy_(cov)
            self.dircov_ema.mul_(1.0 - ema_tau).add_(cov, alpha=ema_tau)

    def anchor_loss(self, lambda_u: float = 0.0, lambda_b: Optional[float] = None) -> th.Tensor:
        if (not self._shaping_enabled) or (lambda_u <= 0.0 and (lambda_b is None or lambda_b <= 0.0)):
            return self.U.sum() * 0.0
        if lambda_b is None:
            lambda_b = lambda_u
        loss = self.U.new_zeros(())
        if lambda_u > 0.0:
            loss = loss + float(lambda_u) * (self.U - self.U_anchor).pow(2).sum()
        if self.has_bias and lambda_b > 0.0:
            loss = loss + float(lambda_b) * (self.b - self.b_anchor).pow(2).sum()
        return loss

    def hoyer_loss(self, lambda_col: float = 0.0, lambda_row: float = 0.0, eps: float = 1e-8) -> th.Tensor:
        if (not self._shaping_enabled) or (lambda_col <= 0.0 and lambda_row <= 0.0):
            return self.U.sum() * 0.0
        U = self.U
        loss = U.new_zeros(())
        if lambda_col > 0.0:
            l1 = U.abs().sum(dim=0)
            l2 = (U.pow(2).sum(dim=0) + eps).sqrt()
            loss = loss + float(lambda_col) * (l1 / l2).sum()
        if lambda_row > 0.0:
            l1 = U.abs().sum(dim=1)
            l2 = (U.pow(2).sum(dim=1) + eps).sqrt()
            loss = loss + float(lambda_row) * (l1 / l2).sum()
        return loss

    def dircov_subspace_loss(
        self,
        lambda_dircov: float = 0.0,
        *,
        normalize_cov: bool = True,
        detach_cov: bool = True,
        eps: float = 1e-8,
    ) -> th.Tensor:
        if (not self._shaping_enabled) or lambda_dircov <= 0.0:
            return self.U.sum() * 0.0
        C = self.dircov_ema
        if detach_cov:
            C = C.detach()
        C = C.to(dtype=self.U.dtype, device=self.U.device)
        C = 0.5 * (C + C.T)
        if normalize_cov:
            C = C / th.trace(C).clamp(min=eps)
        energy = th.trace(self.U.T @ C @ self.U)
        return -float(lambda_dircov) * energy / float(max(1, self.muscle_group_num))

    def orthogonal_loss(
        self,
        lambda_ortho: float = 0.0,
        *,
        normalize_columns: bool = True,
        offdiag_only: bool = True,
        eps: float = 1e-8,
    ) -> th.Tensor:
        if (not self._shaping_enabled) or lambda_ortho <= 0.0:
            return self.U.sum() * 0.0
        U = self.U
        if normalize_columns:
            U = U / U.norm(dim=0, keepdim=True).clamp(min=eps)
        gram = U.T @ U
        diff = gram - th.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        if offdiag_only:
            diff = diff - th.diag_embed(th.diagonal(diff))
        return float(lambda_ortho) * diff.pow(2).mean()

    def regularization_terms(
        self,
        *,
        lambda_anchor: float = 0.0,
        lambda_anchor_b: Optional[float] = None,
        lambda_hoyer_col: float = 0.0,
        lambda_hoyer_row: float = 0.0,
        lambda_dircov: float = 0.0,
        lambda_ortho: float = 0.0,
        hoyer_eps: float = 1e-8,
        dircov_normalize_cov: bool = True,
        ortho_normalize_columns: bool = True,
        ortho_offdiag_only: bool = True,
    ) -> Dict[str, th.Tensor]:
        return {
            "anchor": self.anchor_loss(lambda_u=lambda_anchor, lambda_b=lambda_anchor_b),
            "hoyer": self.hoyer_loss(lambda_col=lambda_hoyer_col, lambda_row=lambda_hoyer_row, eps=hoyer_eps),
            "dircov": self.dircov_subspace_loss(
                lambda_dircov=lambda_dircov,
                normalize_cov=dircov_normalize_cov,
                detach_cov=True,
                eps=hoyer_eps,
            ),
            "ortho": self.orthogonal_loss(
                lambda_ortho=lambda_ortho,
                normalize_columns=ortho_normalize_columns,
                offdiag_only=ortho_offdiag_only,
                eps=hoyer_eps,
            ),
        }


class DevynQSACLayer(DevynQLayerBase):
    def __init__(
        self,
        *args,
        learnable_tau: bool = True,
        init_tau: float = 2.0,
        per_channel_tau: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.learnable_tau = bool(learnable_tau)
        self.per_channel_tau = bool(per_channel_tau)
        shape = (self.muscle_dims,) if self.per_channel_tau else ()
        tau0 = th.full(shape, float(init_tau), dtype=th.float32)
        self.tau_raw = nn.Parameter(tau0.clone(), requires_grad=self.learnable_tau)

    def _tau(self) -> th.Tensor:
        return th.nn.functional.softplus(self.tau_raw) + self._tiny

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = super().forward(x)
        tau = self._tau().to(dtype=y.dtype, device=y.device)
        return th.tanh(tau * y)
