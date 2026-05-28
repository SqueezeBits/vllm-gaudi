import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM


class HpuGemma4ForCausalLM(Gemma4ForCausalLM):
    """HPU registry wrapper for upstream Gemma4 text models."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, *args, **kwargs):
        # HPUModelRunner supplies padded [batch, seq] token/position tensors,
        # while Gemma4's per-layer embedding path indexes as flattened
        # [num_tokens, num_layers, dim].  Flatten before entering upstream
        # Gemma4 so PLE dimensions stay layer-major rather than seq-major.
        if input_ids is not None and input_ids.dim() > 1:
            input_ids = input_ids.reshape(-1)
        if positions is not None and positions.dim() > 1:
            positions = positions.reshape(-1)
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is not None and inputs_embeds.dim() > 2:
            kwargs["inputs_embeds"] = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        return super().forward(input_ids, positions, *args, **kwargs)
