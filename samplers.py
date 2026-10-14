from typing import Any, Callable, Dict, List, Optional
import torch
from tqdm import trange

import comfy.k_diffusion.sampling
import comfy.samplers
from .math_core import (
    apply_history_smoothing,
    apply_lipschitz_stabilization,
    compute_phase_lead_velocity,
    compute_velocity,
)


@torch.no_grad()
def sample_flow_phase_corrected(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    disable: Optional[bool] = None,
    phase_strength: float = 1.0,
    history_smoothing: float = 0.25,
) -> torch.Tensor:
    extra_args = {} if extra_args is None else extra_args
    x_current: torch.Tensor = x.clone()
    num_steps: int = len(sigmas) - 1
    s_in: torch.Tensor = x.new_ones([x.shape[0]])
    velocity_history: List[torch.Tensor] = []
    v_prev: Optional[torch.Tensor] = None
    dt_prev: Optional[float] = None

    for step_idx in trange(num_steps, disable=disable):
        sigma_curr: torch.Tensor = sigmas[step_idx]
        sigma_next: torch.Tensor = sigmas[step_idx + 1]
        dt: float = float((sigma_next - sigma_curr).item())

        denoised: torch.Tensor = model(
            x_current, sigma_curr * s_in, **extra_args
        )
        v_curr: torch.Tensor = compute_velocity(
            x_current, sigma_curr * s_in, denoised
        )

        v_smooth: torch.Tensor = apply_history_smoothing(
            v_curr, velocity_history, history_smoothing
        )
        v_lead: torch.Tensor = compute_phase_lead_velocity(
            v_smooth, v_prev, dt, dt_prev, phase_strength
        )

        x_current = x_current + dt * v_lead

        velocity_history.append(v_curr.detach().clone())
        if len(velocity_history) > 6:
            velocity_history.pop(0)

        v_prev = v_curr
        dt_prev = dt

        if callback is not None:
            callback(
                {
                    "x": x_current,
                    "i": step_idx,
                    "sigma": sigma_curr,
                    "sigma_hat": sigma_curr,
                    "denoised": denoised,
                }
            )

    return x_current


@torch.no_grad()
def sample_flow_trapezoidal_pc(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    disable: Optional[bool] = None,
    phase_strength: float = 0.5,
    history_smoothing: float = 0.2,
) -> torch.Tensor:
    extra_args = {} if extra_args is None else extra_args
    x_current: torch.Tensor = x.clone()
    num_steps: int = len(sigmas) - 1
    s_in: torch.Tensor = x.new_ones([x.shape[0]])
    velocity_history: List[torch.Tensor] = []
    v_prev: Optional[torch.Tensor] = None
    dt_prev: Optional[float] = None

    for step_idx in trange(num_steps, disable=disable):
        sigma_curr: torch.Tensor = sigmas[step_idx]
        sigma_next: torch.Tensor = sigmas[step_idx + 1]
        dt: float = float((sigma_next - sigma_curr).item())

        denoised_curr: torch.Tensor = model(
            x_current, sigma_curr * s_in, **extra_args
        )
        v_curr: torch.Tensor = compute_velocity(
            x_current, sigma_curr * s_in, denoised_curr
        )

        v_smooth: torch.Tensor = apply_history_smoothing(
            v_curr, velocity_history, history_smoothing
        )
        v_lead: torch.Tensor = compute_phase_lead_velocity(
            v_smooth, v_prev, dt, dt_prev, phase_strength
        )

        x_pred: torch.Tensor = x_current + dt * v_lead

        denoised_next: torch.Tensor = model(
            x_pred, sigma_next * s_in, **extra_args
        )
        v_next: torch.Tensor = compute_velocity(
            x_pred, sigma_next * s_in, denoised_next
        )
        v_next_stab: torch.Tensor = apply_lipschitz_stabilization(
            v_next, v_lead
        )

        v_trapezoidal: torch.Tensor = 0.5 * (v_lead + v_next_stab)
        x_current = x_current + dt * v_trapezoidal

        velocity_history.append(v_curr.detach().clone())
        if len(velocity_history) > 6:
            velocity_history.pop(0)

        v_prev = v_curr
        dt_prev = dt

        if callback is not None:
            callback(
                {
                    "x": x_current,
                    "i": step_idx,
                    "sigma": sigma_curr,
                    "sigma_hat": sigma_curr,
                    "denoised": denoised_curr,
                }
            )

    return x_current


FLOW_SAMPLER_FUNCS: Dict[str, Callable] = {
    "flow_phase_corrected": sample_flow_phase_corrected,
    "flow_trapezoidal_pc": sample_flow_trapezoidal_pc,
}


def register_custom_samplers() -> None:
    for sampler_name, sampler_func in FLOW_SAMPLER_FUNCS.items():
        if sampler_name not in comfy.samplers.SAMPLER_NAMES:
            comfy.samplers.SAMPLER_NAMES.append(sampler_name)
        setattr(
            comfy.k_diffusion.sampling,
            f"sample_{sampler_name}",
            sampler_func,
        )
