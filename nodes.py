from typing import Any, Dict, Optional, Tuple
import torch

import comfy.samplers
from .integrator import FlowEditor, FlowInverter, FlowTrajectory


class _InversionSampler:
    def __init__(self, inverter: FlowInverter) -> None:
        self.inverter: FlowInverter = inverter
        self.trajectory: Optional[FlowTrajectory] = None

    def sample(
        self,
        model_wrap: Any,
        sigmas: torch.Tensor,
        extra_args: Dict[str, Any],
        callback: Any,
        noise: torch.Tensor,
        latent_image: torch.Tensor,
        denoise_mask: Any,
        disable_pbar: bool,
    ) -> torch.Tensor:
        def model_eval(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            s_in: torch.Tensor = x.new_ones([x.shape[0]])
            return model_wrap(x, sigma * s_in, **extra_args)

        samples, self.trajectory = self.inverter.invert(
            model_fn=model_eval,
            source_latent=latent_image,
            sigmas_inversion=sigmas,
            callback=None,
        )
        return samples


class _EditSampler:
    def __init__(
        self,
        editor: FlowEditor,
        trajectory: Optional[FlowTrajectory],
        model_source_guider: Optional[Any],
        mask: Optional[torch.Tensor],
    ) -> None:
        self.editor: FlowEditor = editor
        self.trajectory: Optional[FlowTrajectory] = trajectory
        self.model_source_guider: Optional[Any] = model_source_guider
        self.mask: Optional[torch.Tensor] = mask

    def sample(
        self,
        model_wrap: Any,
        sigmas: torch.Tensor,
        extra_args: Dict[str, Any],
        callback: Any,
        noise: torch.Tensor,
        latent_image: torch.Tensor,
        denoise_mask: Any,
        disable_pbar: bool,
    ) -> torch.Tensor:
        def model_target_eval(
            x: torch.Tensor, sigma: torch.Tensor
        ) -> torch.Tensor:
            s_in: torch.Tensor = x.new_ones([x.shape[0]])
            return model_wrap(x, sigma * s_in, **extra_args)

        model_source_eval = None
        if self.model_source_guider is not None:
            def model_source_eval(
                x: torch.Tensor, sigma: torch.Tensor
            ) -> torch.Tensor:
                s_in: torch.Tensor = x.new_ones([x.shape[0]])
                return self.model_source_guider(x, sigma * s_in, **extra_args)

        start_state: torch.Tensor = (
            noise if noise is not None else latent_image
        )
        return self.editor.sample_edit(
            model_target_fn=model_target_eval,
            initial_latent=start_state,
            sigmas_generation=sigmas,
            model_source_fn=model_source_eval,
            trajectory=self.trajectory,
            mask=self.mask,
            callback=None,
        )


class FlowInversionNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "steps": (
                    "INT",
                    {"default": 25, "min": 1, "max": 200, "step": 1},
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "scheduler": (
                    comfy.samplers.SCHEDULER_NAMES,
                    {"default": "normal"},
                ),
                "phase_correction": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "history_smoothing": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_corrector_iters": (
                    "INT",
                    {"default": 1, "min": 0, "max": 5, "step": 1},
                ),
            }
        }

    RETURN_TYPES: Tuple[str, str] = ("LATENT", "FLOW_TRAJECTORY")
    RETURN_NAMES: Tuple[str, str] = ("noise_latent", "flow_trajectory")
    FUNCTION: str = "invert"
    CATEGORY: str = "sampling/flow_inversion"

    def invert(
        self,
        model: Any,
        latent: Dict[str, Any],
        positive: Any,
        negative: Any,
        steps: int,
        cfg: float,
        scheduler: str,
        phase_correction: float,
        history_smoothing: float,
        max_corrector_iters: int,
    ) -> Tuple[Dict[str, Any], FlowTrajectory]:
        latent_tensor: torch.Tensor = latent["samples"]
        inverter: FlowInverter = FlowInverter(
            phase_correction=phase_correction,
            history_smoothing=history_smoothing,
            max_corrector_iters=max_corrector_iters,
        )
        sampler_obj: _InversionSampler = _InversionSampler(inverter=inverter)

        sigmas_standard: torch.Tensor = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, steps
        )
        sigmas_inversion: torch.Tensor = sigmas_standard.flip(0)

        guider: comfy.samplers.CFGGuider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        inverted_latent: torch.Tensor = guider.sample(
            noise=latent_tensor,
            latent_image=latent_tensor,
            sampler=sampler_obj,
            sigmas=sigmas_inversion,
            denoise_mask=None,
            callback=None,
            disable_pbar=False,
            seed=0,
        )

        out_latent: Dict[str, Any] = {
            k: v for k, v in latent.items() if k != "samples"
        }
        out_latent["samples"] = inverted_latent
        flow_trajectory: FlowTrajectory = (
            sampler_obj.trajectory
            if sampler_obj.trajectory is not None
            else FlowTrajectory()
        )
        return (out_latent, flow_trajectory)


class FlowInversionSamplerNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "noise_latent": ("LATENT",),
                "steps": (
                    "INT",
                    {"default": 25, "min": 1, "max": 200, "step": 1},
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "structure_fidelity": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "phase_compensation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "history_smoothing": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "scheduler": (
                    comfy.samplers.SCHEDULER_NAMES,
                    {"default": "normal"},
                ),
            },
            "optional": {
                "flow_trajectory": ("FLOW_TRAJECTORY",),
                "source_positive": ("CONDITIONING",),
                "source_negative": ("CONDITIONING",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES: Tuple[str] = ("LATENT",)
    RETURN_NAMES: Tuple[str] = ("latent",)
    FUNCTION: str = "sample_edit"
    CATEGORY: str = "sampling/flow_inversion"

    def sample_edit(
        self,
        model: Any,
        positive: Any,
        negative: Any,
        noise_latent: Dict[str, Any],
        steps: int,
        cfg: float,
        structure_fidelity: float,
        phase_compensation: float,
        history_smoothing: float,
        scheduler: str,
        flow_trajectory: Optional[FlowTrajectory] = None,
        source_positive: Optional[Any] = None,
        source_negative: Optional[Any] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any]]:
        latent_tensor: torch.Tensor = noise_latent["samples"]
        editor: FlowEditor = FlowEditor(
            phase_compensation=phase_compensation,
            history_smoothing=history_smoothing,
            structure_fidelity=structure_fidelity,
        )

        source_guider: Optional[comfy.samplers.CFGGuider] = None
        if source_positive is not None:
            source_guider = comfy.samplers.CFGGuider(model)
            source_guider.set_conds(source_positive, source_negative)
            source_guider.set_cfg(1.0)

        edit_sampler_obj: _EditSampler = _EditSampler(
            editor=editor,
            trajectory=flow_trajectory,
            model_source_guider=source_guider,
            mask=mask,
        )

        sigmas: torch.Tensor = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, steps
        )

        guider: comfy.samplers.CFGGuider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        edited_latent: torch.Tensor = guider.sample(
            noise=latent_tensor,
            latent_image=latent_tensor,
            sampler=edit_sampler_obj,
            sigmas=sigmas,
            denoise_mask=None,
            callback=None,
            disable_pbar=False,
            seed=0,
        )

        out_latent: Dict[str, Any] = {
            k: v for k, v in noise_latent.items() if k != "samples"
        }
        out_latent["samples"] = edited_latent
        return (out_latent,)


class FlowInversionImg2ImgNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "target_positive": ("CONDITIONING",),
                "target_negative": ("CONDITIONING",),
                "source_positive": ("CONDITIONING",),
                "source_negative": ("CONDITIONING",),
                "steps": (
                    "INT",
                    {"default": 25, "min": 1, "max": 200, "step": 1},
                ),
                "cfg_inversion": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "cfg_edit": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "structure_fidelity": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "phase_correction": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "history_smoothing": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "scheduler": (
                    comfy.samplers.SCHEDULER_NAMES,
                    {"default": "normal"},
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES: Tuple[str, str] = ("LATENT", "LATENT")
    RETURN_NAMES: Tuple[str, str] = ("edited_latent", "inverted_noise")
    FUNCTION: str = "execute"
    CATEGORY: str = "sampling/flow_inversion"

    def execute(
        self,
        model: Any,
        latent: Dict[str, Any],
        target_positive: Any,
        target_negative: Any,
        source_positive: Any,
        source_negative: Any,
        steps: int,
        cfg_inversion: float,
        cfg_edit: float,
        structure_fidelity: float,
        phase_correction: float,
        history_smoothing: float,
        scheduler: str,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inversion_node: FlowInversionNode = FlowInversionNode()
        noise_latent, trajectory = inversion_node.invert(
            model=model,
            latent=latent,
            positive=source_positive,
            negative=source_negative,
            steps=steps,
            cfg=cfg_inversion,
            scheduler=scheduler,
            phase_correction=phase_correction,
            history_smoothing=history_smoothing,
            max_corrector_iters=1,
        )

        sampler_node: FlowInversionSamplerNode = FlowInversionSamplerNode()
        (edited_latent,) = sampler_node.sample_edit(
            model=model,
            positive=target_positive,
            negative=target_negative,
            noise_latent=noise_latent,
            steps=steps,
            cfg=cfg_edit,
            structure_fidelity=structure_fidelity,
            phase_compensation=phase_correction,
            history_smoothing=history_smoothing,
            scheduler=scheduler,
            flow_trajectory=trajectory,
            source_positive=source_positive,
            source_negative=source_negative,
            mask=mask,
        )

        return (edited_latent, noise_latent)
