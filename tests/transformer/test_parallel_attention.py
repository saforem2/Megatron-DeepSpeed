import pytest

pytest.importorskip("transformer_engine", reason="transformer_engine is required for transformer tests")

import torch

from megatron.core.transformer.parallel_attention import ParallelAttention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for attention tests"
)


@pytest.fixture
def parallel_attention(transformer_config):
    return ParallelAttention(transformer_config)


@pytest.fixture
def checkpointed_parallel_attention(transformer_config):
    transformer_config.recompute_granularity = "selective"
    return ParallelAttention(transformer_config)


class TestParallelAttention:
    def test_constructor(self, parallel_attention):
        assert isinstance(parallel_attention, ParallelAttention)
        assert parallel_attention.layer_number == 1

        num_weights = sum([p.numel() for p in parallel_attention.parameters()])
        assert num_weights == 624

    def test_cpu_forward(self, parallel_attention):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self, parallel_attention):
        config = parallel_attention.config
        sequence_length = 32
        micro_batch_size = 2

        parallel_attention.cuda()

        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, parallel_attention.config.hidden_size), device="cuda"
        )
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool, device="cuda")

        output, bias = parallel_attention(hidden_states, attention_mask)

        assert config.recompute_granularity is None
        assert output.shape == (sequence_length, micro_batch_size, config.hidden_size)
        assert bias.shape[0] == config.hidden_size

    def test_checkpointed_gpu_forward(self, checkpointed_parallel_attention):
        config = checkpointed_parallel_attention.config
        sequence_length = 32
        micro_batch_size = 2

        checkpointed_parallel_attention.cuda()

        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, checkpointed_parallel_attention.config.hidden_size), device="cuda"
        )
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool, device="cuda")

        output, bias = checkpointed_parallel_attention(hidden_states, attention_mask)

        assert config.recompute_granularity == "selective"
        assert output.shape == (sequence_length, micro_batch_size, config.hidden_size)
        assert bias.shape[0] == config.hidden_size
