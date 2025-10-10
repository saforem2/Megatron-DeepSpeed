import pytest

pytest.importorskip("transformer_engine", reason="transformer_engine is required for transformer tests")

import torch

from megatron.core.transformer.core_attention import CoreAttention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for core attention tests"
)


@pytest.fixture
def core_attention(transformer_config):
    return CoreAttention(transformer_config)


class TestCoreAttention:
    def test_constructor(self, core_attention):
        assert isinstance(core_attention, CoreAttention)
        assert core_attention.layer_number == 1

        num_weights = sum([p.numel() for p in core_attention.parameters()])
        assert num_weights == 0

    def test_cpu_forward(self, core_attention):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self, core_attention):
        core_attention.cuda()
        config = core_attention.config
        sequence_length = 32
        micro_batch_size = 2
        query_layer = torch.ones(
            (
                sequence_length,
                micro_batch_size,
                config.num_attention_heads,
                config.hidden_size // config.num_attention_heads,
            ),
            device="cuda",
        )

        key_layer = torch.ones_like(query_layer, device="cuda")
        value_layer = torch.ones_like(query_layer, device="cuda")
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool, device="cuda")

        context_layer = core_attention(
            query_layer=query_layer, key_layer=key_layer, value_layer=value_layer, attention_mask=attention_mask
        )

        assert context_layer.shape == (sequence_length, micro_batch_size, config.hidden_size)
        assert context_layer.device.type == "cuda"
        assert context_layer.dtype == torch.float32
