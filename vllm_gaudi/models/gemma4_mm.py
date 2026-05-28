import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4_mm import (
    Gemma4DummyInputsBuilder,
    Gemma4ForConditionalGeneration,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class HpuGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    """HPU registry wrapper for upstream Gemma4 multimodal models."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, *args, **kwargs):
        # Same HPU runner adaptation as the text wrapper: Gemma4 PLE code is
        # token-flat, but HPU runner batches token_ids as [batch, seq].
        if input_ids is not None and input_ids.dim() > 1:
            input_ids = input_ids.reshape(-1)
        if positions is not None and positions.dim() > 1:
            positions = positions.reshape(-1)
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is not None and inputs_embeds.dim() > 2:
            kwargs["inputs_embeds"] = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        return super().forward(input_ids, positions, *args, **kwargs)
