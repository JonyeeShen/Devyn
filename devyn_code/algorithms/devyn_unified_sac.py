from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3 import SAC
from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor, create_mlp
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import Actor, LOG_STD_MAX, LOG_STD_MIN, SACPolicy

from devyn_code.algorithms.common.devynq_layer import DevynQSACLayer


class DevynSACActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        group_num: int = None,
        bias: bool = False,
        bias_init: Optional[np.ndarray] = None,
        init_std: float = 1.0,
        shaping_start_steps: Optional[int] = None,
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        assert group_num is not None and group_num > 0
        self.group_num = int(group_num)

        muscle_dims = get_action_dim(self.action_space)
        self.devyn_layer = DevynQSACLayer(
            muscle_dims=muscle_dims,
            group_num=self.group_num,
            shaping_start_steps=shaping_start_steps,
            bias=bias,
            bias_init=bias_init,
            init_std=init_std,
        )

        action_dim = self.group_num
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)

        if use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim,
                full_std=full_std,
                use_expln=use_expln,
                learn_features=True,
                squash_output=True,
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim,
                latent_sde_dim=last_layer_dim,
                log_std_init=log_std_init,
            )
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        a = self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)
        return self.devyn_layer(a)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        actions, log_prob = self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
        return self.devyn_layer(actions), log_prob


class DevynSACPolicy(SACPolicy):
    def __init__(
        self,
        *args,
        group_num: int,
        bias: bool = False,
        bias_init: Optional[np.ndarray] = None,
        init_std: float = 1.0,
        shaping_start_steps: Optional[int] = None,
        **kwargs,
    ):
        self.group_num = int(group_num)
        self.bias = bias
        self.bias_init = bias_init
        self.init_std = float(init_std)
        self.shaping_start_steps = shaping_start_steps

        if "observation_space" in kwargs:
            obs_space = kwargs.get("observation_space")
        elif len(args) > 0:
            obs_space = args[0]
        else:
            obs_space = None

        is_dict_space = isinstance(obs_space, spaces.Dict)
        if not is_dict_space and hasattr(obs_space, "spaces"):
            try:
                is_dict_space = isinstance(getattr(obs_space, "spaces"), dict)
            except Exception:
                is_dict_space = False
        if is_dict_space:
            kwargs.setdefault("features_extractor_class", CombinedExtractor)

        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        self.actor_kwargs.update(
            {
                "group_num": self.group_num,
                "bias": self.bias,
                "bias_init": self.bias_init,
                "init_std": self.init_std,
                "shaping_start_steps": self.shaping_start_steps,
            }
        )
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DevynSACActor(**actor_kwargs).to(self.device)


class DevynSAC(SAC):
    """Unified Devyn backbone with optional Q and R modules.

    - Q module: direction-covariance shaping (enabled by devyn_enable_q)
    - R module: periodic random perturbation (enabled by devyn_enable_r)
    """

    _DIRCOV_CENTER = True
    _DIRCOV_NORMALIZE_COV = True
    _DIRCOV_NORMALIZE_ROWS = True
    _ORTHO_NORMALIZE_COLUMNS = True
    _ORTHO_OFFDIAG_ONLY = True
    _HOYER_EPS = 1e-8

    _ADAPTIVE_ENABLED = True
    _UNCERTAINTY_EMA_TAU = 0.01
    _UNCERTAINTY_EPS = 1e-6
    _ANCHOR_ADAPT_GAIN = 1.5
    _EMA_TAU_ADAPT_GAIN = 2.0
    _DIRCOV_ADAPT_GAIN = 1.5
    _ANCHOR_SCALE_MIN = 1.0
    _ANCHOR_SCALE_MAX = 5.0
    _EMA_TAU_SCALE_MIN = 0.25
    _DIRCOV_SCALE_MIN = 0.25

    def __init__(self, *args, **kwargs):
        self.devyn_enable_q = bool(kwargs.pop("devyn_enable_q", False))
        self.devyn_enable_r = bool(kwargs.pop("devyn_enable_r", False))

        self.devyn_ema_tau = float(kwargs.pop("devyn_ema_tau", kwargs.pop("nosyn_ema_tau", 0.08)))
        self.devyn_l2sp_lambda = float(kwargs.pop("devyn_l2sp_lambda", kwargs.pop("nosyn_l2sp_lambda", 3.0)))
        self.devyn_l2sp_lambda_b = float(
            kwargs.pop("devyn_l2sp_lambda_b", kwargs.pop("nosyn_l2sp_lambda_b", self.devyn_l2sp_lambda))
        )
        self.devyn_anchor_T = int(kwargs.pop("devyn_anchor_T", kwargs.pop("nosyn_anchor_T", 1000)))

        self.devyn_hoyer_lambda = float(kwargs.pop("devyn_hoyer_lambda", kwargs.pop("nosyn_hoyer_lambda", 0.1)))
        self.devyn_hoyer_row_lambda = float(
            kwargs.pop("devyn_hoyer_row_lambda", kwargs.pop("nosyn_hoyer_row_lambda", self.devyn_hoyer_lambda))
        )
        self.dual_hoyer = bool(kwargs.pop("dual_hoyer", True))

        self.devyn_dircov_lambda = float(kwargs.pop("devyn_dircov_lambda", kwargs.pop("nosyn_dircov_lambda", 0.03)))
        self.devyn_dircov_ema_tau = float(
            kwargs.pop("devyn_dircov_ema_tau", kwargs.pop("nosyn_dircov_ema_tau", 0.005))
        )
        self.devyn_dircov_update_every = int(
            kwargs.pop("devyn_dircov_update_every", kwargs.pop("nosyn_dircov_update_every", 4))
        )
        self.devyn_dircov_warmup_steps = int(
            kwargs.pop("devyn_dircov_warmup_steps", kwargs.pop("nosyn_dircov_warmup_steps", 300000))
        )
        self.devyn_dircov_update_until = int(
            kwargs.pop("devyn_dircov_update_until", kwargs.pop("nosyn_dircov_update_until", -1))
        )

        self.devyn_ortho_lambda = float(kwargs.pop("devyn_ortho_lambda", kwargs.pop("nosyn_ortho_lambda", 0.001)))

        self.devyn_perturb_every = int(kwargs.pop("devyn_perturb_every", kwargs.pop("nosyn_perturb_every", 0)))
        self.devyn_perturb_mode = str(kwargs.pop("devyn_perturb_mode", kwargs.pop("nosyn_perturb_mode", "col")))
        self.devyn_perturb_ratio = float(kwargs.pop("devyn_perturb_ratio", kwargs.pop("nosyn_perturb_ratio", 0.1)))
        self.devyn_perturb_shrink = float(kwargs.pop("devyn_perturb_shrink", kwargs.pop("nosyn_perturb_shrink", 0.05)))
        self.devyn_perturb_noise = float(kwargs.pop("devyn_perturb_noise", kwargs.pop("nosyn_perturb_noise", 0.05)))
        self.devyn_perturb_until = int(kwargs.pop("devyn_perturb_until", kwargs.pop("nosyn_perturb_until", -1)))
        self.devyn_perturb_eps = float(kwargs.pop("devyn_perturb_eps", kwargs.pop("nosyn_perturb_eps", 1e-4)))

        self._devyn_step = 0
        self._uncertainty_ema = 1.0
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        layer = self.policy.actor.devyn_layer
        with th.no_grad():
            layer.U_ema.copy_(layer.U.detach())
            layer.U_anchor.copy_(layer.U.detach())
            if getattr(layer, "has_bias", False):
                layer.b_ema.copy_(layer.b.detach())
                layer.b_anchor.copy_(layer.b.detach())

    def _compute_uncertainty_scalar(
        self,
        current_q_values: Tuple[th.Tensor, ...],
        target_q_values: th.Tensor,
    ) -> float:
        with th.no_grad():
            q_stack = th.stack([q.detach() for q in current_q_values], dim=0)
            q_mean = q_stack.mean(dim=0)
            td_abs = (target_q_values.detach() - q_mean).abs()
            td_var = float(td_abs.var(unbiased=False).item())
            if q_stack.shape[0] >= 2:
                twin_disagreement = float((q_stack[0] - q_stack[1]).abs().mean().item())
            else:
                twin_disagreement = 0.0
            return td_var + twin_disagreement

    def _adaptive_scales(self, uncertainty_scalar: float) -> Dict[str, float]:
        if not self._ADAPTIVE_ENABLED:
            return {"anchor_scale": 1.0, "ema_tau_scale": 1.0, "dircov_scale": 1.0}

        tau = self._UNCERTAINTY_EMA_TAU
        self._uncertainty_ema = (1.0 - tau) * self._uncertainty_ema + tau * float(uncertainty_scalar)
        ratio = float(uncertainty_scalar) / max(self._UNCERTAINTY_EPS, self._uncertainty_ema)
        excess = max(0.0, ratio - 1.0)

        anchor_scale = min(
            self._ANCHOR_SCALE_MAX,
            max(self._ANCHOR_SCALE_MIN, 1.0 + self._ANCHOR_ADAPT_GAIN * excess),
        )
        ema_tau_scale = max(self._EMA_TAU_SCALE_MIN, 1.0 / (1.0 + self._EMA_TAU_ADAPT_GAIN * excess))
        dircov_scale = max(self._DIRCOV_SCALE_MIN, 1.0 / (1.0 + self._DIRCOV_ADAPT_GAIN * excess))
        return {
            "anchor_scale": anchor_scale,
            "ema_tau_scale": ema_tau_scale,
            "dircov_scale": dircov_scale,
        }

    def _maybe_update_direction_cov(
        self,
        layer: DevynQSACLayer,
        base_actor_loss: th.Tensor,
        actions_pi: th.Tensor,
        dircov_scale: float,
    ) -> Optional[float]:
        if not self.devyn_enable_q:
            return None
        if self.devyn_dircov_lambda <= 0.0:
            return None
        if self.num_timesteps < self.devyn_dircov_warmup_steps:
            return None
        if self.devyn_dircov_update_every <= 0:
            return None
        if (self._devyn_step % self.devyn_dircov_update_every) != 0:
            return None
        if self.devyn_dircov_update_until > 0 and self._devyn_step > self.devyn_dircov_update_until:
            return None

        grad_a = th.autograd.grad(
            outputs=base_actor_loss,
            inputs=actions_pi,
            grad_outputs=None,
            retain_graph=True,
            create_graph=False,
            only_inputs=True,
            allow_unused=False,
        )[0]
        layer.update_direction_cov(
            grad_a,
            ema_tau=self.devyn_dircov_ema_tau * dircov_scale,
            center=self._DIRCOV_CENTER,
            normalize_rows=self._DIRCOV_NORMALIZE_ROWS,
            eps=self._HOYER_EPS,
        )
        with th.no_grad():
            return float(th.trace(layer.dircov_ema).item())

    def _maybe_perturb_devyn(self) -> None:
        if not self.devyn_enable_r:
            return
        if self.devyn_perturb_every <= 0:
            return
        if self._devyn_step <= 0:
            return
        if (self._devyn_step % self.devyn_perturb_every) != 0:
            return
        if self.devyn_perturb_until > 0 and self._devyn_step > self.devyn_perturb_until:
            return

        layer = getattr(self.policy.actor, "devyn_layer", None)
        if layer is None:
            return

        with th.no_grad():
            U = layer.U
            m_dim, g_dim = U.shape

            mode = self.devyn_perturb_mode
            ratio = float(np.clip(self.devyn_perturb_ratio, 0.0, 1.0))
            shrink = float(self.devyn_perturb_shrink)
            noise_coeff = float(self.devyn_perturb_noise)
            elem_eps = float(self.devyn_perturb_eps)

            def _perturb_submatrix(sub: th.Tensor) -> th.Tensor:
                elem_scale = sub.abs()
                if elem_eps > 0.0:
                    elem_scale = elem_scale.clamp(min=elem_eps)
                noise = th.randn_like(sub) * (noise_coeff * elem_scale)
                prob_minus = th.full_like(sub, 0.5)
                prob_minus = th.where(sub > 1.0, 0.5 * sub, prob_minus)
                prob_minus = th.where(sub < -1.0, 1.0 - 0.5 * sub.abs(), prob_minus)
                prob_minus = prob_minus.clamp(0.0, 1.0)
                mask = th.rand_like(sub) < prob_minus
                pos = (1.0 - shrink) * sub + noise
                neg = (1.0 + shrink) * sub - noise
                return th.where(mask, pos, neg)

            if mode == "col":
                k = max(1, int(round(g_dim * ratio)))
                idx = th.randperm(g_dim, device=U.device)[:k]
                U[:, idx] = _perturb_submatrix(U[:, idx])
            elif mode == "row":
                k = max(1, int(round(m_dim * ratio)))
                idx = th.randperm(m_dim, device=U.device)[:k]
                U[idx, :] = _perturb_submatrix(U[idx, :])
            elif mode == "both":
                k_col = max(1, int(round(g_dim * ratio)))
                idx_col = th.randperm(g_dim, device=U.device)[:k_col]
                U[:, idx_col] = _perturb_submatrix(U[:, idx_col])

                k_row = max(1, int(round(m_dim * ratio)))
                idx_row = th.randperm(m_dim, device=U.device)[:k_row]
                U[idx_row, :] = _perturb_submatrix(U[idx_row, :])
            else:
                raise ValueError(f"Unknown devyn_perturb_mode={mode}")

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        anchor_losses, hoyer_losses, dircov_losses, ortho_losses = [], [], [], []
        dircov_trace_stats = []

        for gradient_step in range(gradient_steps):
            self._devyn_step += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            self.policy.actor.devyn_layer.maybe_enable_shaping(self.num_timesteps)

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = self.critic_target(replay_data.next_observations, next_actions)
                next_q_values = th.cat(next_q_values, dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(
                th.nn.functional.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            q_values_pi = self.critic(replay_data.observations, actions_pi)
            min_qf_pi = th.cat(q_values_pi, dim=1).min(dim=1, keepdim=True)[0]
            base_actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            uncertainty_scalar = self._compute_uncertainty_scalar(current_q_values, target_q_values)
            adaptive = self._adaptive_scales(uncertainty_scalar)

            layer = self.policy.actor.devyn_layer
            trace_val = self._maybe_update_direction_cov(layer, base_actor_loss, actions_pi, adaptive["dircov_scale"])
            if trace_val is not None:
                dircov_trace_stats.append(trace_val)

            effective_dircov_lambda = self.devyn_dircov_lambda * adaptive["dircov_scale"] if self.devyn_enable_q else 0.0
            reg_terms = layer.regularization_terms(
                lambda_anchor=self.devyn_l2sp_lambda * adaptive["anchor_scale"],
                lambda_anchor_b=self.devyn_l2sp_lambda_b * adaptive["anchor_scale"],
                lambda_hoyer_col=self.devyn_hoyer_lambda,
                lambda_hoyer_row=self.devyn_hoyer_row_lambda if self.dual_hoyer else 0.0,
                lambda_dircov=effective_dircov_lambda,
                lambda_ortho=self.devyn_ortho_lambda,
                hoyer_eps=self._HOYER_EPS,
                dircov_normalize_cov=self._DIRCOV_NORMALIZE_COV,
                ortho_normalize_columns=self._ORTHO_NORMALIZE_COLUMNS,
                ortho_offdiag_only=self._ORTHO_OFFDIAG_ONLY,
            )

            actor_loss = (
                base_actor_loss
                + reg_terms["anchor"]
                + reg_terms["hoyer"]
                + reg_terms["dircov"]
                + reg_terms["ortho"]
            )

            actor_losses.append(base_actor_loss.item())
            anchor_losses.append(reg_terms["anchor"].item())
            hoyer_losses.append(reg_terms["hoyer"].item())
            dircov_losses.append(reg_terms["dircov"].item())
            ortho_losses.append(reg_terms["ortho"].item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            with th.no_grad():
                tau_eff = self.devyn_ema_tau * adaptive["ema_tau_scale"]
                layer.U_ema.mul_(1.0 - tau_eff).add_(layer.U.detach(), alpha=tau_eff)
                if getattr(layer, "has_bias", False):
                    layer.b_ema.mul_(1.0 - tau_eff).add_(layer.b.detach(), alpha=tau_eff)

            self._maybe_perturb_devyn()

            with th.no_grad():
                if self.devyn_anchor_T > 0 and (self._devyn_step % self.devyn_anchor_T) == 0:
                    layer.U_anchor.copy_(layer.U.detach())
                    if getattr(layer, "has_bias", False):
                        layer.b_anchor.copy_(layer.b.detach())

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/devyn_anchor_loss", np.mean(anchor_losses))
        self.logger.record("train/devyn_hoyer_loss", np.mean(hoyer_losses))
        self.logger.record("train/devyn_dircov_loss", np.mean(dircov_losses))
        self.logger.record("train/devyn_ortho_loss", np.mean(ortho_losses))
        self.logger.record("train/devyn_enable_q", float(self.devyn_enable_q))
        self.logger.record("train/devyn_enable_r", float(self.devyn_enable_r))
        if len(dircov_trace_stats) > 0:
            self.logger.record("train/devyn_dircov_trace", np.mean(dircov_trace_stats))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(self, *args: Any, **kwargs: Any):
        try:
            self.policy.actor.devyn_layer.maybe_enable_shaping(self.num_timesteps)
        except Exception:
            pass
        return super().learn(*args, **kwargs)
