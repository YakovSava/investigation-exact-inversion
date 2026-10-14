from typing import Any, Dict, Optional, Tuple
import torch

import comfy.samplers
from .integrator import FlowEditor, FlowInverter, FlowTrajectory
from .math_core import is_active_mask


class FlowInversionGuider(comfy.samplers.CFGGuider):
    def __init__(self, model_patcher: Any) -> None:
        super().__init__(model_patcher)
        self.cfg_inversion: float = 1.0
        self.cfg_target: float = 3.5

    def set_flow_conds(
        self,
        src_pos: Any,
        src_neg: Any,
        tgt_pos: Any,
        tgt_neg: Any,
    ) -> None:
        self.inner_set_conds(
            {
                "source_positive": src_pos,
                "source_negative": src_neg,
                "target_positive": tgt_pos,
                "target_negative": tgt_neg,
            }
        )

    def set_flow_cfgs(
        self,
        cfg_inversion: float,
        cfg_target: float,
    ) -> None:
        self.cfg_inversion = cfg_inversion
        self.cfg_target = cfg_target

    def predict_noise(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        model_options: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        opts: Dict[str, Any] = {} if model_options is None else model_options
        branch: str = opts.get("flow_branch", "target")
        pos: Any = self.conds.get(f"{branch}_positive", None)
        neg: Any = self.conds.get(f"{branch}_negative", None)
        cfg: float = (
            self.cfg_inversion if branch == "source" else self.cfg_target
        )
        return comfy.samplers.sampling_function(
            self.inner_model,
            x,
            timestep,
            neg,
            pos,
            cfg,
            model_options=opts,
            seed=seed,
        )


class _UnifiedFlowKSampler:
    def __init__(
        self,
        inverter: FlowInverter,
        editor: FlowEditor,
        mask: Optional[torch.Tensor],
    ) -> None:
        self.inverter: FlowInverter = inverter
        self.editor: FlowEditor = editor
        self.mask: Optional[torch.Tensor] = mask
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
        def evaluate_branch(
            x: torch.Tensor,
            sigma: torch.Tensor,
            branch: str,
        ) -> torch.Tensor:
            s_in: torch.Tensor = x.new_ones([x.shape[0]])
            options: Dict[str, Any] = extra_args.setdefault(
                "model_options", {}
            )
            options["flow_branch"] = branch
            return model_wrap(x, sigma * s_in, **extra_args)

        def source_eval(
            x: torch.Tensor,
            s: torch.Tensor,
        ) -> torch.Tensor:
            return evaluate_branch(x, s, "source")

        def target_eval(
            x: torch.Tensor,
            s: torch.Tensor,
        ) -> torch.Tensor:
            return evaluate_branch(x, s, "target")

        sigmas_inversion: torch.Tensor = sigmas.flip(0)
        start_state: torch.Tensor = latent_image

        inverted_noise, trajectory = self.inverter.invert(
            model_fn=source_eval,
            source_latent=start_state,
            sigmas_inversion=sigmas_inversion,
            callback=None,
        )
        self.trajectory = trajectory

        source_steer = (
            source_eval if self.editor.structure_fidelity > 0.0 else None
        )
        return self.editor.sample_edit(
            model_target_fn=target_eval,
            initial_latent=inverted_noise,
            sigmas_generation=sigmas,
            model_source_fn=source_steer,
            trajectory=trajectory,
            mask=self.mask,
            callback=None,
        )


class FlowInversionKSampler:
    empty_image: torch.Tensor = torch.zeros(
        (1, 1, 1, 3), dtype=torch.float32
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 200, "step": 1},
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "cfg_inversion": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "structure_fidelity": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "phase_correction": (
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
                "source_positive": ("CONDITIONING",),
                "source_negative": ("CONDITIONING",),
                "optional_vae": ("VAE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES: Tuple[str, str, str, str, str, str] = (
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "VAE",
        "IMAGE",
    )
    RETURN_NAMES: Tuple[str, str, str, str, str, str] = (
        "MODEL",
        "CONDITIONING+",
        "CONDITIONING-",
        "LATENT",
        "VAE",
        "IMAGE",
    )
    FUNCTION: str = "sample"
    CATEGORY: str = "sampling/flow_inversion"

    def sample(
        self,
        model: Any,
        positive: Any,
        negative: Any,
        latent_image: Dict[str, Any],
        seed: int,
        steps: int,
        cfg: float,
        cfg_inversion: float,
        denoise: float,
        structure_fidelity: float,
        phase_correction: float,
        history_smoothing: float,
        scheduler: str,
        source_positive: Optional[Any] = None,
        source_negative: Optional[Any] = None,
        optional_vae: Optional[Any] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, Any, Any, Dict[str, Any], Any, torch.Tensor]:
        src_pos: Any = (
            source_positive if source_positive is not None else positive
        )
        src_neg: Any = (
            source_negative if source_negative is not None else negative
        )

        inverter: FlowInverter = FlowInverter(
            phase_correction=phase_correction,
            history_smoothing=history_smoothing,
            max_corrector_iters=0,
        )
        editor: FlowEditor = FlowEditor(
            phase_compensation=phase_correction,
            history_smoothing=history_smoothing,
            structure_fidelity=structure_fidelity,
        )

        active_mask: Optional[torch.Tensor] = (
            mask if is_active_mask(mask) else None
        )
        sampler_obj: _UnifiedFlowKSampler = _UnifiedFlowKSampler(
            inverter=inverter,
            editor=editor,
            mask=active_mask,
        )

        sigmas: torch.Tensor = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, steps
        )
        if denoise < 1.0:
            slice_start: int = int(round((len(sigmas) - 1) * (1.0 - denoise)))
            sigmas = sigmas[slice_start:]

        guider: FlowInversionGuider = FlowInversionGuider(model)
        guider.set_flow_conds(src_pos, src_neg, positive, negative)
        guider.set_flow_cfgs(cfg_inversion=cfg_inversion, cfg_target=cfg)

        latent_tensor: torch.Tensor = latent_image["samples"]
        edited_tensor: torch.Tensor = guider.sample(
            noise=latent_tensor,
            latent_image=latent_tensor,
            sampler=sampler_obj,
            sigmas=sigmas,
            denoise_mask=None,
            callback=None,
            disable_pbar=False,
            seed=seed,
        )

        out_latent: Dict[str, Any] = {
            k: v for k, v in latent_image.items() if k != "samples"
        }
        out_latent["samples"] = edited_tensor

        decoded_images: torch.Tensor = self.empty_image
        if optional_vae is not None:
            decoded_images = optional_vae.decode(edited_tensor)

        return (
            model,
            positive,
            negative,
            out_latent,
            optional_vae,
            decoded_images,
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
                    {"default": 0, "min": 0, "max": 5, "step": 1},
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
        inverter: FlowInverter = FlowInverter(
            phase_correction=phase_correction,
            history_smoothing=history_smoothing,
            max_corrector_iters=max_corrector_iters,
        )
        trajectory: FlowTrajectory = FlowTrajectory()

        class _StandaloneInverter:
            def sample(
                inner_self,
                model_wrap: Any,
                sigmas: torch.Tensor,
                extra_args: Dict[str, Any],
                callback: Any,
                noise: torch.Tensor,
                latent_image: torch.Tensor,
                denoise_mask: Any,
                disable_pbar: bool,
            ) -> torch.Tensor:
                nonlocal trajectory
                def model_eval(
                    x: torch.Tensor,
                    sigma: torch.Tensor,
                ) -> torch.Tensor:
                    s_in: torch.Tensor = x.new_ones([x.shape[0]])
                    return model_wrap(x, sigma * s_in, **extra_args)

                samples, trajectory = inverter.invert(
                    model_fn=model_eval,
                    source_latent=latent_image,
                    sigmas_inversion=sigmas,
                    callback=None,
                )
                return samples

        sigmas_standard: torch.Tensor = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, steps
        )
        sigmas_inversion: torch.Tensor = sigmas_standard.flip(0)

        guider: comfy.samplers.CFGGuider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        latent_tensor: torch.Tensor = latent["samples"]
        inverted_latent: torch.Tensor = guider.sample(
            noise=latent_tensor,
            latent_image=latent_tensor,
            sampler=_StandaloneInverter(),
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
        return (out_latent, trajectory)


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
        active_mask: Optional[torch.Tensor] = (
            mask if is_active_mask(mask) else None
        )

        if flow_trajectory is None:
            ksampler: FlowInversionKSampler = FlowInversionKSampler()
            res = ksampler.sample(
                model=model,
                positive=positive,
                negative=negative,
                latent_image=noise_latent,
                seed=0,
                steps=steps,
                cfg=cfg,
                cfg_inversion=1.0,
                denoise=1.0,
                structure_fidelity=structure_fidelity,
                phase_correction=phase_compensation,
                history_smoothing=history_smoothing,
                scheduler=scheduler,
                source_positive=source_positive,
                source_negative=source_negative,
                optional_vae=None,
                mask=active_mask,
            )
            return (res[3],)

        editor: FlowEditor = FlowEditor(
            phase_compensation=phase_compensation,
            history_smoothing=history_smoothing,
            structure_fidelity=structure_fidelity,
        )

        class _StandaloneEditor:
            def sample(
                inner_self,
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
                    x: torch.Tensor,
                    sigma: torch.Tensor,
                ) -> torch.Tensor:
                    s_in: torch.Tensor = x.new_ones([x.shape[0]])
                    return model_wrap(x, sigma * s_in, **extra_args)

                model_source_eval = None
                if source_positive is not None:
                    src_guider: comfy.samplers.CFGGuider = (
                        comfy.samplers.CFGGuider(model)
                    )
                    src_guider.set_conds(source_positive, source_negative)
                    src_guider.set_cfg(1.0)

                    def model_source_eval(
                        x: torch.Tensor,
                        sigma: torch.Tensor,
                    ) -> torch.Tensor:
                        s_in: torch.Tensor = x.new_ones([x.shape[0]])
                        return src_guider(x, sigma * s_in, **extra_args)

                return editor.sample_edit(
                    model_target_fn=model_target_eval,
                    initial_latent=latent_image,
                    sigmas_generation=sigmas,
                    model_source_fn=model_source_eval,
                    trajectory=flow_trajectory,
                    mask=active_mask,
                    callback=None,
                )

        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, steps
        )

        guider: comfy.samplers.CFGGuider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        latent_tensor: torch.Tensor = noise_latent["samples"]
        edited_latent: torch.Tensor = guider.sample(
            noise=latent_tensor,
            latent_image=latent_tensor,
            sampler=_StandaloneEditor(),
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
        ksampler: FlowInversionKSampler = FlowInversionKSampler()
        res = ksampler.sample(
            model=model,
            positive=target_positive,
            negative=target_negative,
            latent_image=latent,
            seed=0,
            steps=steps,
            cfg=cfg_edit,
            cfg_inversion=cfg_inversion,
            denoise=1.0,
            structure_fidelity=structure_fidelity,
            phase_correction=phase_correction,
            history_smoothing=history_smoothing,
            scheduler=scheduler,
            source_positive=source_positive,
            source_negative=source_negative,
            optional_vae=None,
            mask=mask,
        )
        return (res[3], latent)
