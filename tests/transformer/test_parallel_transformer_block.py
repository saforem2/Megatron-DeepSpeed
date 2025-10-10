import pytest

pytest.importorskip("transformer_engine", reason="transformer_engine is required for transformer tests")

import torch

from megatron.core.transformer.parallel_transformer_layer import ParallelTransformerLayer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for transformer block tests"
)


@pytest.fixture
def transformer_layer(transformer_config):
    return ParallelTransformerLayer(transformer_config)


def test_transformer_block_forward(transformer_layer, transformer_config):
    transformer_layer.cuda()
    hidden_states = torch.ones((32, 2, transformer_config.hidden_size), device="cuda")
    attention_mask = torch.ones((1, 1, 32, 32), dtype=bool, device="cuda")
    output, attention_bias = transformer_layer(hidden_states, attention_mask)
    assert output.shape == (32, 2, transformer_config.hidden_size)
    assert attention_bias.shape[0] == transformer_config.hidden_size
