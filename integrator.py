from dataclasses import dataclass, field
from typing import Callable, List, Optional
import torch

from .math_core import (
    apply_history_smoothing,
    apply_lipschitz_stabilization,
    blend_inpainting_latents,
    compute_phase_lead_velocity,
    compute_velocity,
)


@dataclass
class FlowTrajectory:
    states: List[torch.Tensor] = field(default_factory=list)
    sigmas: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float32)
    )
    velocities: List[torch.Tensor] = field(default_factory=list)
    displacements: List[torch.Tensor] = field(default_factory=list)
    source_latent: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float32)
    )


class FlowInverter:
    def __init__(
        self,
        phase_correction: float = 1.0,
        history_smoothing: float = 0.35,
        max_corrector_iters: int = 1,
    ) -> None:
        self.phase_correction: float = phase_correction
        self.history_smoothing: float = history_smoothing
        self.max_corrector_iters: int = max_corrector_iters

    def invert(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        source_latent: torch.Tensor,
        sigmas_inversion: torch.Tensor,
        callback: Optional[Callable[[int, torch.Tensor, int], None]] = None,
    ) -> tuple[torch.Tensor, FlowTrajectory]:
        trajectory: FlowTrajectory = FlowTrajectory(
            source_latent=source_latent.detach().clone(),
            sigmas=sigmas_inversion.detach().clone(),
        )
        x_current: torch.Tensor = source_latent.clone()
        trajectory.states.append(x_current.clone())
        velocity_history: List[torch.Tensor] = []
        num_steps: int = len(sigmas_inversion) - 1
        v_prev: Optional[torch.Tensor] = None
        dt_prev: Optional[float] = None

        for step_idx in range(num_steps):
            sigma_curr: torch.Tensor = sigmas_inversion[step_idx]
            sigma_next: torch.Tensor = sigmas_inversion[step_idx + 1]
            dt: float = float((sigma_next - sigma_curr).item())

            sigma_eval: torch.Tensor = sigma_curr.expand(
                x_current.shape[0]
            ).to(x_current.device)
            denoised_curr: torch.Tensor = model_fn(x_current, sigma_eval)
            v_curr: torch.Tensor = compute_velocity(
                x_current, sigma_eval, denoised_curr
            )

            v_smooth: torch.Tensor = apply_history_smoothing(
                v_curr, velocity_history, self.history_smoothing
            )
            v_lead: torch.Tensor = compute_phase_lead_velocity(
                v_smooth, v_prev, dt, dt_prev, self.phase_correction
            )

            x_next_candidate: torch.Tensor = x_current + dt * v_lead

            for _ in range(self.max_corrector_iters):
                sigma_next_eval: torch.Tensor = sigma_next.expand(
                    x_next_candidate.shape[0]
                ).to(x_next_candidate.device)
                denoised_next: torch.Tensor = model_fn(
                    x_next_candidate, sigma_next_eval
                )
                v_next: torch.Tensor = compute_velocity(
                    x_next_candidate, sigma_next_eval, denoised_next
                )
                v_next_stabilized: torch.Tensor = (
                    apply_lipschitz_stabilization(v_next, v_lead)
                )
                v_trapezoidal: torch.Tensor = 0.5 * (
                    v_lead + v_next_stabilized
                )
                x_next_candidate = x_current + dt * v_trapezoidal

            displacement: torch.Tensor = x_next_candidate - x_current
            trajectory.velocities.append(v_curr.detach().clone())
            trajectory.displacements.append(displacement.detach().clone())
            velocity_history.append(v_curr.detach().clone())
            if len(velocity_history) > 6:
                velocity_history.pop(0)

            v_prev = v_curr
            dt_prev = dt
            x_current = x_next_candidate
            trajectory.states.append(x_current.clone())

            if callback is not None:
                callback(step_idx, x_current, num_steps)

        return x_current, trajectory


class FlowEditor:
    def __init__(
        self,
        phase_compensation: float = 1.0,
        history_smoothing: float = 0.25,
        structure_fidelity: float = 0.75,
    ) -> None:
        self.phase_compensation: float = phase_compensation
        self.history_smoothing: float = history_smoothing
        self.structure_fidelity: float = structure_fidelity

    def sample_edit(
        self,
        model_target_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        initial_latent: torch.Tensor,
        sigmas_generation: torch.Tensor,
        model_source_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        trajectory: Optional[FlowTrajectory] = None,
        mask: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, torch.Tensor, int], None]] = None,
    ) -> torch.Tensor:
        x_current: torch.Tensor = initial_latent.clone()
        velocity_history: List[torch.Tensor] = []
        num_steps: int = len(sigmas_generation) - 1
        v_prev: Optional[torch.Tensor] = None
        dt_prev: Optional[float] = None

        has_trajectory: bool = (
            trajectory is not None
            and len(trajectory.displacements) >= num_steps
        )

        for step_idx in range(num_steps):
            sigma_curr: torch.Tensor = sigmas_generation[step_idx]
            sigma_next: torch.Tensor = sigmas_generation[step_idx + 1]
            dt: float = float((sigma_next - sigma_curr).item())

            sigma_eval: torch.Tensor = sigma_curr.expand(
                x_current.shape[0]
            ).to(x_current.device)
            denoised_target: torch.Tensor = model_target_fn(
                x_current, sigma_eval
            )
            v_target: torch.Tensor = compute_velocity(
                x_current, sigma_eval, denoised_target
            )

            traj_idx: int = num_steps - 1 - step_idx

            if has_trajectory and trajectory is not None:
                src_disp: torch.Tensor = trajectory.displacements[
                    traj_idx
                ].to(x_current.device)

                if (
                    model_source_fn is not None
                    and self.structure_fidelity > 0.0
                ):
                    denoised_source: torch.Tensor = model_source_fn(
                        x_current, sigma_eval
                    )
                    v_source: torch.Tensor = compute_velocity(
                        x_current, sigma_eval, denoised_source
                    )
                    v_delta: torch.Tensor = v_target - v_source
                    v_delta_smooth: torch.Tensor = apply_history_smoothing(
                        v_delta, velocity_history, self.history_smoothing
                    )
                    v_delta_lead: torch.Tensor = (
                        compute_phase_lead_velocity(
                            v_delta_smooth,
                            v_prev,
                            dt,
                            dt_prev,
                            self.phase_compensation,
                        )
                    )

                    step_update: torch.Tensor = (
                        -self.structure_fidelity * src_disp
                        + (1.0 - self.structure_fidelity) * (dt * v_target)
                        + self.structure_fidelity * (dt * v_delta_lead)
                    )
                    v_steered = v_delta
                else:
                    v_smooth: torch.Tensor = apply_history_smoothing(
                        v_target, velocity_history, self.history_smoothing
                    )
                    v_lead: torch.Tensor = compute_phase_lead_velocity(
                        v_smooth, v_prev, dt, dt_prev, self.phase_compensation
                    )
                    step_update = (
                        -self.structure_fidelity * src_disp
                        + (1.0 - self.structure_fidelity) * (dt * v_lead)
                    )
                    v_steered = v_target
            else:
                v_smooth = apply_history_smoothing(
                    v_target, velocity_history, self.history_smoothing
                )
                v_lead = compute_phase_lead_velocity(
                    v_smooth, v_prev, dt, dt_prev, self.phase_compensation
                )
                step_update = dt * v_lead
                v_steered = v_target

            x_next: torch.Tensor = x_current + step_update

            if (
                mask is not None
                and has_trajectory
                and trajectory is not None
                and len(trajectory.states) > traj_idx
            ):
                source_reference: torch.Tensor = trajectory.states[
                    traj_idx
                ].to(x_next.device)
                x_next = blend_inpainting_latents(
                    x_next, source_reference, mask
                )

            velocity_history.append(v_steered.detach().clone())
            if len(velocity_history) > 6:
                velocity_history.pop(0)

            v_prev = v_steered
            dt_prev = dt
            x_current = x_next

            if callback is not None:
                callback(step_idx, x_current, num_steps)

        return x_current
