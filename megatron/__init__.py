# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from .deepspeed_compat import ensure_deepspeed_stub

ensure_deepspeed_stub()

from .global_vars import get_args, get_retro_args
from .global_vars import get_current_global_batch_size
from .global_vars import get_num_microbatches
from .global_vars import get_signal_handler
from .global_vars import update_num_microbatches
from .global_vars import get_tokenizer
from .global_vars import get_tensorboard_writer
from .global_vars import get_wandb_writer
from .global_vars import get_adlr_autoresume
from .global_vars import get_timers

from .utils import (print_rank_0, is_last_rank, print_rank_last, is_rank_0, is_aml)


def initialize_megatron(*args, **kwargs):
    from .initialize import initialize_megatron as _initialize_megatron

    return _initialize_megatron(*args, **kwargs)
