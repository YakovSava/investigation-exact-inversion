import math
from typing import Tuple
import torch

from .integrator import FlowEditor, FlowInverter, FlowTrajectory
from .math_core import (
    apply_history_smoothing,
    apply_lipschitz_stabilization,
    blend_inpainting_latents,
    compute_phase_lead_velocity,
    compute_velocity,
    reconstruct_denoised,
)


def mock_linear_flow_model(
    x: torch.Tensor,
    sigma: torch.Tensor,
    x0_target: torch.Tensor,
) -> torch.Tensor:
    return x0_target.expand_as(x)


def mock_curved_flow_model(
    x: torch.Tensor,
    sigma: torch.Tensor,
    x0_target: torch.Tensor,
) -> torch.Tensor:
    sigma_broadcast: torch.Tensor = sigma.view(-1, 1, 1, 1)
    curvature: torch.Tensor = 0.15 * torch.sin(x) * (1.0 - sigma_broadcast)
    return x0_target.expand_as(x) + curvature


def test_velocity_and_reconstruction() -> None:
    x: torch.Tensor = torch.randn(2, 4, 32, 32)
    x0: torch.Tensor = torch.randn(2, 4, 32, 32)
    sigma: torch.Tensor = torch.tensor([0.5, 0.5])
    v: torch.Tensor = compute_velocity(x, sigma, x0)
    x0_rec: torch.Tensor = reconstruct_denoised(x, sigma, v)
    assert torch.allclose(x0, x0_rec, atol=1e-5)


def test_phase_lead_and_smoothing() -> None:
    v_curr: torch.Tensor = torch.ones(2, 4, 16, 16) * 2.0
    v_prev: torch.Tensor = torch.ones(2, 4, 16, 16) * 1.0
    v_lead: torch.Tensor = compute_phase_lead_velocity(
        v_current=v_curr,
        v_prev=v_prev,
        dt=0.1,
        dt_prev=0.1,
        strength=1.0,
    )
    expected_v_lead: torch.Tensor = torch.ones(2, 4, 16, 16) * 2.5
    assert torch.allclose(v_lead, expected_v_lead, atol=1e-5)

    history: list[torch.Tensor] = [
        torch.ones(2, 4, 16, 16) * 1.0,
        torch.ones(2, 4, 16, 16) * 1.5,
    ]
    v_smooth: torch.Tensor = apply_history_smoothing(
        v_current=v_curr,
        velocity_history=history,
        smoothing_factor=0.3,
    )
    assert v_smooth.shape == v_curr.shape
    assert not torch.isnan(v_smooth).any()


def test_lipschitz_stabilization() -> None:
    v_ref: torch.Tensor = torch.ones(2, 4, 16, 16)
    v_cand_exploding: torch.Tensor = torch.ones(2, 4, 16, 16) * 50.0
    v_stab: torch.Tensor = apply_lipschitz_stabilization(
        v_candidate=v_cand_exploding,
        v_reference=v_ref,
        max_relative_variation=2.0,
    )
    assert torch.norm(v_stab - v_ref) < torch.norm(v_cand_exploding - v_ref)


def test_inpainting_blending() -> None:
    source: torch.Tensor = torch.zeros(1, 4, 16, 16)
    target: torch.Tensor = torch.ones(1, 4, 16, 16)
    mask: torch.Tensor = torch.zeros(1, 16, 16)
    mask[:, :8, :] = 1.0
    blended: torch.Tensor = blend_inpainting_latents(
        target_latent=target,
        source_latent=source,
        mask=mask,
        blur_kernel_size=1,
    )
    assert torch.allclose(blended[:, :, :8, :], target[:, :, :8, :])
    assert torch.allclose(blended[:, :, 8:, :], source[:, :, 8:, :])


def test_inversion_reconstruction_roundtrip() -> Tuple[float, float]:
    torch.manual_seed(42)
    clean_latent: torch.Tensor = torch.randn(1, 4, 32, 32)
    sigmas_gen: torch.Tensor = torch.linspace(1.0, 0.0, 21)
    sigmas_inv: torch.Tensor = sigmas_gen.flip(0)

    inverter: FlowInverter = FlowInverter(
        phase_correction=1.0,
        history_smoothing=0.3,
        max_corrector_iters=1,
    )

    def model_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return mock_curved_flow_model(x, sigma, clean_latent)

    inverted_noise, trajectory = inverter.invert(
        model_fn=model_fn,
        source_latent=clean_latent,
        sigmas_inversion=sigmas_inv,
    )

    editor: FlowEditor = FlowEditor(
        phase_compensation=1.0,
        history_smoothing=0.25,
        structure_fidelity=1.0,
    )

    reconstructed_latent: torch.Tensor = editor.sample_edit(
        model_target_fn=model_fn,
        initial_latent=inverted_noise,
        sigmas_generation=sigmas_gen,
        model_source_fn=model_fn,
        trajectory=trajectory,
    )

    mse_error: float = float(
        torch.mean((clean_latent - reconstructed_latent) ** 2).item()
    )
    max_error: float = float(
        torch.max(torch.abs(clean_latent - reconstructed_latent)).item()
    )
    assert mse_error < 1e-8
    return mse_error, max_error


def run_all_tests() -> None:
    test_velocity_and_reconstruction()
    test_phase_lead_and_smoothing()
    test_lipschitz_stabilization()
    test_inpainting_blending()
    mse, max_err = test_inversion_reconstruction_roundtrip()
    print(
        f"All tests passed. Inversion MSE: {mse:.6e}, Max Diff: {max_err:.6e}"
    )


if __name__ == "__main__":
    run_all_tests()
