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
    epsilon: float = 1e-3,
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


def is_active_mask(mask: Optional[torch.Tensor]) -> bool:
    if mask is None:
        return False
    if mask.numel() == 0:
        return False
    if torch.count_nonzero(mask) == 0:
        return False
    if float(mask.max().item()) <= 1e-4:
        return False
    return True


def blend_inpainting_latents(
    target_latent: torch.Tensor,
    source_latent: torch.Tensor,
    mask: Optional[torch.Tensor],
    blur_kernel_size: int = 5,
) -> torch.Tensor:
    if not is_active_mask(mask) or mask is None:
        return target_latent
    mask_prepared: torch.Tensor = mask.to(
        device=target_latent.device, dtype=target_latent.dtype
    )
    if mask_prepared.ndim == 2:
        mask_prepared = mask_prepared.unsqueeze(0).unsqueeze(0)
    elif mask_prepared.ndim == 3:
        mask_prepared = mask_prepared.unsqueeze(1)
    spatial_shape: Tuple[int, int] = (
        target_latent.shape[-2],
        target_latent.shape[-1],
    )
    if mask_prepared.shape[-2:] != spatial_shape:
        mask_prepared = F.interpolate(
            mask_prepared,
            size=spatial_shape,
            mode="bilinear",
            align_corners=False,
        )
    if blur_kernel_size > 1 and blur_kernel_size % 2 == 1:
        pad: int = blur_kernel_size // 2
        kernel_2d: torch.Tensor = torch.ones(
            1,
            1,
            blur_kernel_size,
            blur_kernel_size,
            device=mask_prepared.device,
            dtype=mask_prepared.dtype,
        ) / float(blur_kernel_size * blur_kernel_size)
        mask_prepared = F.conv2d(mask_prepared, kernel_2d, padding=pad)
    mask_clamped: torch.Tensor = torch.clamp(mask_prepared, 0.0, 1.0)
    mask_expanded: torch.Tensor = (
        mask_clamped.unsqueeze(2)
        if target_latent.ndim == 5
        else mask_clamped
    )
    return (
        (1.0 - mask_expanded) * source_latent
        + mask_expanded * target_latent
    )


def compute_differential_mask(
    v_target: torch.Tensor,
    v_source: torch.Tensor,
    sensitivity: float = 4.0,
) -> torch.Tensor:
    v_diff: torch.Tensor = v_target - v_source
    energy: torch.Tensor = torch.mean(v_diff ** 2, dim=1, keepdim=True)
    flat_energy: torch.Tensor = energy.reshape(-1)
    if flat_energy.numel() == 0:
        return torch.zeros_like(energy)
    max_energy: float = float(torch.max(flat_energy).item())
    if max_energy < 1e-5:
        return torch.zeros_like(energy)
    k: int = max(1, int(flat_energy.numel() * 0.25))
    bg_floor: float = float(torch.kthvalue(flat_energy, k).values.item())
    threshold: float = max(bg_floor * 2.5, 1e-4)
    return torch.sigmoid(sensitivity * (energy / threshold - 1.0))


def extract_low_frequency(
    delta: torch.Tensor,
    kernel_size: int = 7,
) -> torch.Tensor:
    pad: int = kernel_size // 2
    if delta.ndim == 4:
        return F.avg_pool2d(
            delta, kernel_size=kernel_size, stride=1, padding=pad
        )
    elif delta.ndim == 5:
        b: int = delta.shape[0]
        c: int = delta.shape[1]
        t: int = delta.shape[2]
        h: int = delta.shape[3]
        w: int = delta.shape[4]
        d_reshaped: torch.Tensor = delta.permute(0, 2, 1, 3, 4).reshape(
            b * t, c, h, w
        )
        low_pass: torch.Tensor = F.avg_pool2d(
            d_reshaped, kernel_size=kernel_size, stride=1, padding=pad
        )
        return low_pass.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return delta


def suppress_parallel_component(
    delta: torch.Tensor,
    reference: torch.Tensor,
    parallel_weight: float = 0.0,
) -> torch.Tensor:
    dims: List[int] = list(range(1, delta.ndim))
    ref_norm_sq: torch.Tensor = torch.sum(
        reference * reference, dim=dims, keepdim=True
    ).clamp_min(1e-12)
    coefficient: torch.Tensor = (
        torch.sum(delta * reference, dim=dims, keepdim=True) / ref_norm_sq
    )
    parallel: torch.Tensor = coefficient * reference
    return delta - (1.0 - parallel_weight) * parallel
