import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F


def append_dims(tensor: torch.Tensor, target_ndim: int) -> torch.Tensor:
    dims_to_add: int = target_ndim - tensor.ndim
    if dims_to_add < 0:
        raise ValueError(
            f"Target ndim {target_ndim} must be >= tensor ndim {tensor.ndim}"
        )
    return tensor[(...,) + (None,) * dims_to_add]


def compute_velocity(
    x: torch.Tensor,
    sigma: torch.Tensor,
    denoised: torch.Tensor,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    sigma_clamped: torch.Tensor = torch.clamp(sigma, min=epsilon)
    sigma_expanded: torch.Tensor = append_dims(sigma_clamped, x.ndim)
    return (x - denoised) / sigma_expanded


def reconstruct_denoised(
    x: torch.Tensor,
    sigma: torch.Tensor,
    velocity: torch.Tensor,
) -> torch.Tensor:
    sigma_expanded: torch.Tensor = append_dims(sigma, x.ndim)
    return x - sigma_expanded * velocity


def compute_phase_lead_velocity(
    v_current: torch.Tensor,
    v_prev: Optional[torch.Tensor],
    dt: float,
    dt_prev: Optional[float],
    strength: float = 1.0,
) -> torch.Tensor:
    if v_prev is None or dt_prev is None or abs(dt_prev) < 1e-7:
        return v_current
    dv_dt: torch.Tensor = (v_current - v_prev) / dt_prev
    phase_correction: torch.Tensor = 0.5 * strength * dt * dv_dt
    return v_current + phase_correction


def apply_history_smoothing(
    v_current: torch.Tensor,
    velocity_history: List[torch.Tensor],
    smoothing_factor: float = 0.3,
) -> torch.Tensor:
    if not velocity_history or smoothing_factor <= 0.0:
        return v_current
    history_length: int = len(velocity_history)
    weights: List[float] = [
        math.exp(-0.5 * (history_length - 1 - idx))
        for idx in range(history_length)
    ]
    total_weight: float = sum(weights)
    normalized_weights: List[float] = [w / total_weight for w in weights]
    v_historical_mean: torch.Tensor = torch.zeros_like(v_current)
    for weight, v_past in zip(normalized_weights, velocity_history):
        v_historical_mean = v_historical_mean + weight * v_past
    clamped_factor: float = min(max(smoothing_factor, 0.0), 0.9)
    return (
        (1.0 - clamped_factor) * v_current
        + clamped_factor * v_historical_mean
    )


def apply_lipschitz_stabilization(
    v_candidate: torch.Tensor,
    v_reference: torch.Tensor,
    max_relative_variation: float = 2.5,
) -> torch.Tensor:
    delta_v: torch.Tensor = v_candidate - v_reference
    ref_norm: torch.Tensor = torch.norm(v_reference, dim=1, keepdim=True)
    delta_norm: torch.Tensor = torch.norm(delta_v, dim=1, keepdim=True)
    threshold: torch.Tensor = max_relative_variation * torch.clamp(
        ref_norm, min=1e-4
    )
    scale: torch.Tensor = torch.where(
        delta_norm > threshold,
        threshold / torch.clamp(delta_norm, min=1e-7),
        torch.ones_like(delta_norm),
    )
    return v_reference + delta_v * scale


def blend_inpainting_latents(
    target_latent: torch.Tensor,
    source_latent: torch.Tensor,
    mask: Optional[torch.Tensor],
    blur_kernel_size: int = 5,
) -> torch.Tensor:
    if mask is None:
        return target_latent
    mask_prepared: torch.Tensor = mask.to(
        device=target_latent.device, dtype=target_latent.dtype
    )
    if mask_prepared.ndim == 2:
        mask_prepared = mask_prepared.unsqueeze(0).unsqueeze(0)
    elif mask_prepared.ndim == 3:
        mask_prepared = mask_prepared.unsqueeze(1)
    if mask_prepared.shape[-2:] != target_latent.shape[-2:]:
        mask_prepared = F.interpolate(
            mask_prepared,
            size=target_latent.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if blur_kernel_size > 1 and blur_kernel_size % 2 == 1:
        pad: int = blur_kernel_size // 2
        kernel_1d: torch.Tensor = torch.ones(
            1, 1, blur_kernel_size, blur_kernel_size,
            device=mask_prepared.device,
            dtype=mask_prepared.dtype,
        ) / float(blur_kernel_size * blur_kernel_size)
        mask_prepared = F.conv2d(mask_prepared, kernel_1d, padding=pad)
    mask_clamped: torch.Tensor = torch.clamp(mask_prepared, 0.0, 1.0)
    return (1.0 - mask_clamped) * source_latent + mask_clamped * target_latent
