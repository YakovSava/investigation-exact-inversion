from typing import Dict, Type
from .nodes import (
    FlowInversionImg2ImgNode,
    FlowInversionNode,
    FlowInversionSamplerNode,
)
from .samplers import register_custom_samplers

register_custom_samplers()

NODE_CLASS_MAPPINGS: Dict[str, Type] = {
    "FlowInversion": FlowInversionNode,
    "FlowInversionSampler": FlowInversionSamplerNode,
    "FlowInversionImg2Img": FlowInversionImg2ImgNode,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "FlowInversion": "Flow Inversion (Phase-Corrected)",
    "FlowInversionSampler": "Flow Inversion Sampler / Edit",
    "FlowInversionImg2Img": "Flow Inversion Img2Img / Inpaint",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
