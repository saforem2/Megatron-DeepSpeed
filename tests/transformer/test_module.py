import pytest
import torch

from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

pytest.importorskip("transformer_engine", reason="transformer_engine is required for transformer tests")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for module tests"
)

DEVICE_CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None


class DummyModule(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.linear = torch.nn.modules.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def megatron_module(transformer_config):
    return DummyModule(config=transformer_config).cuda()


class TestMegatronModule:
    def test_megatron_module(self, megatron_module):
        assert megatron_module
        assert megatron_module.config.hidden_size == 12
        assert megatron_module.config.ffn_hidden_size == 48
        assert megatron_module.linear.weight.dtype == torch.float32

        x = torch.ones((2, 2), device="cuda")
        assert megatron_module(x).dtype == torch.float32


class TestFloat16Module:
    def test_fp16_module(self, transformer_config, megatron_module):
        transformer_config.fp16 = True
        fp16_module = Float16Module(config=transformer_config, module=megatron_module)

        assert fp16_module
        assert fp16_module.config.hidden_size == 12
        assert fp16_module.config.ffn_hidden_size == 48
        assert fp16_module.module.linear.weight.dtype == torch.float16

        x = torch.ones((2, 2), device="cuda")
        assert fp16_module(x).dtype == torch.float32

    @pytest.mark.skipif(
        not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 8,
        reason="bfloat16 is not supported on this device",
    )
    def test_bf16_module(self, transformer_config, megatron_module):
        transformer_config.bf16 = True
        bf16_module = Float16Module(config=transformer_config, module=megatron_module)

        assert bf16_module
        assert bf16_module.config.hidden_size == 12
        assert bf16_module.config.ffn_hidden_size == 48
        assert bf16_module.module.linear.weight.dtype == torch.bfloat16

        x = torch.ones((2, 2), device="cuda")
        assert bf16_module(x).dtype == torch.float32
