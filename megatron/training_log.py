# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

from enum import Enum

# from deepspeed.accelerator import get_accelerator
# from deepspeed.compression.compress import redundancy_clean
import torch
import os
import logging

from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_wandb_writer
from megatron import get_num_microbatches
from megatron.core import mpu

# from megatron import is_rank_0, print_rank_0
# from megatron import print_rank_last
# from megatron.arguments import core_transformer_config_from_args
# from megatron.checkpointing import load_checkpoint
# from megatron.checkpointing import save_checkpoint
# from megatron.core import mpu, tensor_parallel
# from megatron.core.enums import ModelType
# from megatron.core.pipeline_parallel import get_forward_backward_func
# from megatron.data.data_samplers import build_pretraining_data_loader
# from megatron.initialize import initialize_megatron
# from megatron.initialize import write_args_to_tensorboard
# from megatron.initialize import set_jit_fusion_options
# from megatron.model import Float16Module
# from megatron.model import GPTModel
# from megatron.model import DistributedDataParallel as LocalDDP
# from megatron.model.transformer import ParallelTransformerLayer
# from megatron.model.vision.knn_monitor import compute_feature_bank
# from megatron.optimizer import get_megatron_optimizer
# from megatron.optimizer_param_scheduler import OptimizerParamScheduler
# from megatron.profiler import on_step_begin, on_step_end, setup_profiler, trigger
# from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import found_kill_switch
import ezpz as ez

# from megatron.utils import calc_params_l2_norm
from megatron.utils import (
    # checkpoint_throughput_calculator,
    report_memory,
    throughput_calculator,
    # update_rotary_pos_emb,
)

try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None
# The earliest we can measure the start time.
# _TRAIN_START_TIME = time.time()


log = logging.getLogger(__name__)


class InteropLoggingTool(Enum):
    TENSORBOARD = 1
    WANDB = 2


RANK: int = ez.get_rank()
LOCAL_RANK: int = ez.get_local_rank()
WORLD_SIZE: int = ez.get_world_size()
DEVICE_TYPE: str = ez.dist.get_torch_device_type()
DEVICE_ID: str = f"{DEVICE_TYPE}:{LOCAL_RANK}"
DEVICE: torch.device = torch.device(DEVICE_TYPE)

log: logging.Logger = logging.getLogger(__name__)
LOG_LEVEL: str = str(os.environ.get("LOG_LEVEL", "INFO")).upper()
log.setLevel(LOG_LEVEL) if RANK == 0 else log.setLevel("CRITICAL")


class interop_tool_logger:
    def __init__(self, tb_writer=None, wandb_writer=None):
        self.tb_writer = tb_writer
        self.wandb_writer = wandb_writer
        self.custom_x_axis = []
        self.custom_y_axis = {}
        self.args = get_args()
        assert self.args is not None
        if not hasattr(self.args, "logger_iteration"):
            self.args.logger_iteration = 1
        assert self.args.logger_iteration is not None

    def is_enabled(self):
        return self.tb_writer or self.wandb_writer

    def add_scalar(
        self,
        key,
        scalar_value,
        step,
        custom_step_name=None,
        tool_list=[InteropLoggingTool.TENSORBOARD, InteropLoggingTool.WANDB],
    ):
        if self.tb_writer and InteropLoggingTool.TENSORBOARD in tool_list:
            self.tb_writer.add_scalar(key, scalar_value, step)

        if (
            wandb is not None
            and self.wandb_writer
            and InteropLoggingTool.WANDB in tool_list
        ):
            assert self.args is not None
            assert self.args.logger_iteration is not None
            if not custom_step_name:
                self.wandb_writer.log({key: scalar_value}, step=step)
                if self.args.logger_iteration < step:
                    # Updating iteration
                    self.args.logger_iteration = step

            else:
                if custom_step_name not in self.custom_x_axis:
                    self.custom_x_axis.append(custom_step_name)
                    wandb.define_metric(custom_step_name)

                if key not in self.custom_y_axis:
                    self.custom_y_axis[key] = custom_step_name
                    wandb.define_metric(key, step_metric=custom_step_name)

                self.wandb_writer.log(
                    {key: scalar_value, custom_step_name: step},
                    step=self.args.logger_iteration,
                )

    def add_scalar_to_tb(self, key, scalar_value, step):
        return self.add_scalar(
            key, scalar_value, step, None, [InteropLoggingTool.TENSORBOARD]
        )

    def add_scalar_to_wandb(self, key, scalar_value, step, custom_step_name=None):
        return self.add_scalar(
            key, scalar_value, step, custom_step_name, [InteropLoggingTool.WANDB]
        )

    def add_images(self, key, img_tensor, step=None):
        if self.tb_writer:
            self.tb_writer.add_images(key, img_tensor, step)

        if wandb is not None and self.wandb_writer:
            self.wandb_writer.log({key: wandb.Image(img_tensor)}, step)


def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
    model=None,
    optimizer=None,
):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = interop_tool_logger(
        tb_writer=get_tensorboard_writer(), wandb_writer=get_wandb_writer()
    )
    x_axis_samples = "Samples"
    x_axis_tokens = "Tokens"
    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = "advanced iterations"
    skipped_iters_key = "skipped iterations"
    nan_iters_key = "nan iterations"
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = (
            total_loss_dict.get(advanced_iters_key, 0) + 1
        )
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = (
        total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    )
    # Update losses and set nan iterations
    got_nan = False
    _zero = torch.tensor([0.0]).to(DEVICE)
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(key, _zero) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(
        got_nan
    )

    # Logging.
    timers_to_log = [
        "forward-backward",
        "forward-compute",
        "backward-compute",
        "batch-generator",
        "forward-recv",
        "forward-send",
        "backward-recv",
        "backward-send",
        "forward-send-forward-recv",
        "forward-send-backward-recv",
        "backward-send-forward-recv",
        "backward-send-backward-recv",
        "forward-backward-send-forward-backward-recv",
        "layernorm-grads-all-reduce",
        "embedding-grads-all-reduce",
        "grads-all-reduce",
        "grads-reduce-scatter",
        "params-all-gather",
        "optimizer-copy-to-main-grad",
        "optimizer-unscale-and-check-inf",
        "optimizer-clip-main-grad",
        "optimizer-count-zeros",
        "optimizer-inner-step",
        "optimizer-copy-main-to-model-params",
        "optimizer",
    ]

    assert args is not None and timers is not None
    # Calculate batch size.
    batch_size = (
        args.micro_batch_size * args.data_parallel_size * get_num_microbatches()
    )

    total_iterations = (
        total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]
    )

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and (
        iteration % args.tensorboard_log_interval == 0
    ):
        timers.write(timers_to_log, writer, iteration, normalizer=total_iterations)  # type: ignore
    if writer.is_enabled() and (iteration % args.tensorboard_log_interval == 0):
        writer.add_scalar(
            "steps-vs-samples/y=steps,x=samples",
            iteration,
            args.consumed_train_samples,
            x_axis_samples,
        )
        writer.add_scalar(
            "steps-vs-samples/y=samples,x=steps", args.consumed_train_samples, iteration
        )
        writer.add_scalar(
            "steps-vs-tokens/y=steps,x=tokens",
            iteration,
            args.consumed_train_tokens,
            x_axis_tokens,
        )
        writer.add_scalar(
            "steps-vs-tokens/y=tokens,x=steps", args.consumed_train_tokens, iteration
        )
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar("learning-rate/learning-rate", learning_rate, iteration)
            writer.add_scalar(
                "learning-rate/learning-rate vs samples",
                learning_rate,
                args.consumed_train_samples,
                x_axis_samples,
            )
            writer.add_scalar(
                "learning-rate/learning-rate vs tokens",
                learning_rate,
                args.consumed_train_tokens,
                x_axis_tokens,
            )
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar("batch-size/batch-size", batch_size, iteration)
            writer.add_scalar(
                "batch-size/batch-size vs samples",
                batch_size,
                args.consumed_train_samples,
                x_axis_samples,
            )
            writer.add_scalar(
                "batch-size/batch-size vs tokens",
                batch_size,
                args.consumed_train_tokens,
                x_axis_tokens,
            )
        for key in loss_dict:
            writer.add_scalar(f"lm-loss-training/{key}", loss_dict[key], iteration)
            writer.add_scalar(
                f"lm-loss-training/{key}" + " vs samples",
                loss_dict[key],
                args.consumed_train_samples,
                x_axis_samples,
            )
            writer.add_scalar(
                f"lm-loss-training/{key}" + " vs tokens",
                loss_dict[key],
                args.consumed_train_tokens,
                x_axis_tokens,
            )
        if args.fp16 and loss_scale and args.log_loss_scale_to_tensorboard:
            writer.add_scalar("loss-scale/loss-scale", loss_scale, iteration)
            writer.add_scalar(
                "loss-scale/loss-scale vs samples",
                loss_scale,
                args.consumed_train_samples,
                x_axis_samples,
            )
            writer.add_scalar(
                "loss-scale/loss-scale vs tokens",
                loss_scale,
                args.consumed_train_tokens,
                x_axis_tokens,
            )
        if args.log_world_size_to_tensorboard:
            writer.add_scalar("world-size/world-size", args.world_size, iteration)
            writer.add_scalar(
                "world-size/world-size vs samples",
                args.world_size,
                args.consumed_train_samples,
                x_axis_samples,
            )
            writer.add_scalar(
                "world-size/world-size vs tokens",
                args.world_size,
                args.consumed_train_tokens,
                x_axis_tokens,
            )
        if grad_norm is not None:
            writer.add_scalar("grad-norm/grad-norm", grad_norm, iteration)
            writer.add_scalar(
                "grad-norm/grad-norm vs samples",
                grad_norm,
                args.consumed_train_samples,
                x_axis_samples,
            )
            writer.add_scalar(
                "grad-norm/grad-norm vs tokens",
                grad_norm,
                args.consumed_train_tokens,
                x_axis_tokens,
            )
        if num_zeros_in_grad is not None:
            writer.add_scalar("num-zeros/num-zeros", num_zeros_in_grad, iteration)
            writer.add_scalar(
                "num-zeros/num-zeros vs samples",
                num_zeros_in_grad,
                args.consumed_train_samples,
                x_axis_samples,
            )
            writer.add_scalar(
                "num-zeros/num-zeros vs tokens",
                num_zeros_in_grad,
                args.consumed_train_tokens,
                x_axis_tokens,
            )
        if params_norm is not None:
            writer.add_scalar("params-norm/params-norm", params_norm, iteration)
            writer.add_scalar(
                "params-norm/params-norm vs samples",
                params_norm,
                args.consumed_train_samples,
                x_axis_samples,
            )
            writer.add_scalar(
                "params-norm/params-norm vs tokens",
                params_norm,
                args.consumed_train_tokens,
                x_axis_tokens,
            )
        if hasattr(args, "actual_seq_length"):
            writer.add_scalar(
                "seqlen/actual_seq_length", args.actual_seq_length, iteration
            )
            writer.add_scalar(
                "seqlen/actual_seq_length vs samples",
                args.actual_seq_length,
                args.consumed_train_samples,
                x_axis_samples,
            )
            writer.add_scalar(
                "seqlen/actual_seq_length vs tokens",
                args.actual_seq_length,
                args.consumed_train_tokens,
                x_axis_tokens,
            )
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            writer.add_scalar(
                "seqlen/curriculum_seqlen", args.curriculum_seqlen, iteration
            )
            writer.add_scalar(
                "seqlen/curriculum_seqlen vs samples",
                args.curriculum_seqlen,
                args.consumed_train_samples,
                x_axis_samples,
            )
            writer.add_scalar(
                "seqlen/curriculum_seqlen vs tokens",
                args.curriculum_seqlen,
                args.consumed_train_tokens,
                x_axis_tokens,
            )
        if args.random_ltd:
            writer.add_scalar(
                "seqlen/random_ltd_reserved_length",
                args.random_ltd_reserved_length,
                iteration,
            )
            writer.add_scalar(
                "seqlen/random_ltd_reserved_length vs samples",
                args.random_ltd_reserved_length,
                args.consumed_train_samples,
                x_axis_samples,
            )
            writer.add_scalar(
                "seqlen/random_ltd_reserved_length vs tokens",
                args.random_ltd_reserved_length,
                args.consumed_train_tokens,
                x_axis_tokens,
            )
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )

    if iteration % args.tensorboard_log_interval == 0:
        # This logging write various optimizer states to tensorboard. This
        # feature may consume extra GPU memory thus is set at false by default.
        if args.log_optimizer_states_to_tensorboard and optimizer is not None:
            opt_stats = [0.0] * 8
            opt_stats_2 = [0.0] * 4

            # TODO(billishyahao): Remove me after bf16_optimizer promotes its state.
            if not hasattr(optimizer, "state"):
                assert hasattr(
                    optimizer, "optimizer"
                ), "Optimizer must have optimizer property."
                optimizer.state = optimizer.optimizer.state

            for _, group in enumerate(optimizer.param_groups):
                for _, param in enumerate(group["params"]):
                    opt_stats[0] += (
                        torch.norm(optimizer.state[param]["exp_avg_sq"]).item()
                    ) ** 2
                    opt_stats[1] += (
                        torch.norm(optimizer.state[param]["exp_avg_sq"].sqrt()).item()
                    ) ** 2
                    opt_stats[2] += (
                        torch.norm(optimizer.state[param]["exp_avg"]).item()
                    ) ** 2
                    opt_stats[3] += (torch.norm(param).item()) ** 2
                    opt_stats[4] += torch.norm(
                        optimizer.state[param]["exp_avg_sq"], p=1
                    ).item()
                    opt_stats[5] += torch.norm(
                        optimizer.state[param]["exp_avg_sq"].sqrt(), p=1
                    ).item()
                    opt_stats[6] += torch.norm(
                        optimizer.state[param]["exp_avg"], p=1
                    ).item()
                    opt_stats[7] += torch.norm(param, p=1).item()
                    opt_stats_2[0] = max(
                        opt_stats_2[0],
                        abs(optimizer.state[param]["exp_avg_sq"].max().item()),
                        abs(optimizer.state[param]["exp_avg_sq"].min().item()),
                    )
                    opt_stats_2[1] = max(
                        opt_stats_2[1],
                        optimizer.state[param]["exp_avg_sq"].sqrt().abs_().max().item(),
                    )
                    opt_stats_2[2] = max(
                        opt_stats_2[2],
                        abs(optimizer.state[param]["exp_avg"].max().item()),
                        abs(optimizer.state[param]["exp_avg"].min().item()),
                    )
                    opt_stats_2[3] = max(
                        opt_stats_2[3], abs(param.max().item()), abs(param.min().item())
                    )
            # print('step {} rank {} before sync opt_stats {}, {}'.format(iteration, torch.distributed.get_rank(), opt_stats_2, opt_stats))
            if args.zero_stage > 0:
                # ZeRO partiions optimizer states
                # opt_stats = get_accelerator().FloatTensor(opt_stats)
                opt_stats = torch.tensor(opt_stats).to(DEVICE)
                torch.distributed.all_reduce(
                    opt_stats, group=mpu.get_sequence_data_parallel_group()
                )
                # opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                opt_stats_2 = torch.tensor(opt_stats_2).to(DEVICE)
                torch.distributed.all_reduce(
                    opt_stats_2,
                    op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_sequence_data_parallel_group(),
                )

            if args.tensor_model_parallel_size > 1:
                opt_stats = torch.tensor(opt_stats).to(DEVICE)
                # opt_stats = get_accelerator().FloatTensor(opt_stats)
                torch.distributed.all_reduce(
                    opt_stats, group=mpu.get_tensor_model_parallel_group()
                )
                # opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                opt_stats_2 = torch.tensor(opt_stats_2).to(DEVICE)
                torch.distributed.all_reduce(
                    opt_stats_2,
                    op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_tensor_model_parallel_group(),
                )

            if args.pipeline_model_parallel_size > 1:
                # opt_stats = get_accelerator().FloatTensor(opt_stats)
                opt_stats = torch.tensor(opt_stats).to(DEVICE)
                torch.distributed.all_reduce(
                    opt_stats, group=mpu.get_pipeline_model_parallel_group()
                )
                # opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                opt_stats_2 = torch.tensor(opt_stats_2).to(DEVICE)
                torch.distributed.all_reduce(
                    opt_stats_2,
                    op=torch.distributed.ReduceOp.MAX,
                    group=mpu.get_pipeline_model_parallel_group(),
                )

            # print('step {} rank {} after sync opt_stats {}, {}'.format(iteration, torch.distributed.get_rank(), opt_stats_2, opt_stats))
            # if writer.is_enabled() and is_last_rank():
            if writer.is_enabled() and RANK == 0:
                writer.add_scalar(
                    "optimizer/variance_l2 vs tokens",
                    opt_stats[0] ** 0.5,
                    args.consumed_train_tokens,
                    x_axis_tokens,
                )
                writer.add_scalar(
                    "optimizer/variance_sqrt_l2 vs tokens",
                    opt_stats[1] ** 0.5,
                    args.consumed_train_tokens,
                    x_axis_tokens,
                )
                writer.add_scalar(
                    "optimizer/momentum_l2 vs tokens",
                    opt_stats[2] ** 0.5,
                    args.consumed_train_tokens,
                    x_axis_tokens,
                )
                writer.add_scalar(
                    "optimizer/weight_l2 vs tokens",
                    opt_stats[3] ** 0.5,
                    args.consumed_train_tokens,
                    x_axis_tokens,
                )
                writer.add_scalar(
                    "optimizer/variance_l1 vs tokens",
                    opt_stats[4],
                    args.consumed_train_tokens,
                    x_axis_tokens,
                )
                writer.add_scalar(
                    "optimizer/variance_sqrt_l1 vs tokens",
                    opt_stats[5],
                    args.consumed_train_tokens,
                    x_axis_tokens,
                )
                writer.add_scalar(
                    "optimizer/momentum_l1 vs tokens",
                    opt_stats[6],
                    args.consumed_train_tokens,
                    x_axis_tokens,
                )
                writer.add_scalar(
                    "optimizer/weight_l1 vs tokens",
                    opt_stats[7],
                    args.consumed_train_tokens,
                    x_axis_tokens,
                )
                writer.add_scalar(
                    "optimizer/variance_abs_max vs tokens",
                    opt_stats_2[0],
                    args.consumed_train_tokens,
                    x_axis_tokens,
                )
                writer.add_scalar(
                    "optimizer/variance_sqrt_abs_max vs tokens",
                    opt_stats_2[1],
                    args.consumed_train_tokens,
                    x_axis_tokens,
                )
                writer.add_scalar(
                    "optimizer/momentum_abs_max vs tokens",
                    opt_stats_2[2],
                    args.consumed_train_tokens,
                    x_axis_tokens,
                )
                writer.add_scalar(
                    "optimizer/weight_abs_max vs tokens",
                    opt_stats_2[3],
                    args.consumed_train_tokens,
                    x_axis_tokens,
                )

                writer.add_scalar(
                    "optimizer/variance_l2", opt_stats[0] ** 0.5, iteration
                )
                writer.add_scalar(
                    "optimizer/variance_sqrt_l2", opt_stats[1] ** 0.5, iteration
                )
                writer.add_scalar(
                    "optimizer/momentum_l2", opt_stats[2] ** 0.5, iteration
                )
                writer.add_scalar("optimizer/weight_l2", opt_stats[3] ** 0.5, iteration)
                writer.add_scalar("optimizer/variance_l1", opt_stats[4], iteration)
                writer.add_scalar("optimizer/variance_sqrt_l1", opt_stats[5], iteration)
                writer.add_scalar("optimizer/momentum_l1", opt_stats[6], iteration)
                writer.add_scalar("optimizer/weight_l1", opt_stats[7], iteration)
                writer.add_scalar(
                    "optimizer/variance_abs_max", opt_stats_2[0], iteration
                )
                writer.add_scalar(
                    "optimizer/variance_sqrt_abs_max", opt_stats_2[1], iteration
                )
                writer.add_scalar(
                    "optimizer/momentum_abs_max", opt_stats_2[2], iteration
                )
                writer.add_scalar("optimizer/weight_abs_max", opt_stats_2[3], iteration)

    if iteration % args.log_interval == 0:
        elapsed_time = timers("interval-time").elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations
        seq_len = args.seq_length
        if hasattr(args, "actual_seq_length"):
            seq_len = args.actual_seq_length
        samples_per_sec, tflops, approx_parameters_in_billions = throughput_calculator(
            model, args, elapsed_time, total_iterations
        )
        samples_per_sec_per_replica = samples_per_sec / args.data_parallel_size
        tokens_per_sec = samples_per_sec * seq_len
        tokens_per_sec_per_replica = tokens_per_sec / args.data_parallel_size
        tokens_per_gpu_per_second = tokens_per_sec / args.world_size
        tokens_per_gpu_per_second_per_replica = (
            tokens_per_gpu_per_second / args.data_parallel_size
        )

        if writer.is_enabled():
            writer.add_scalar_to_wandb(
                "throughput/iteration-time", elapsed_time_per_iteration, iteration
            )  # 1000 ms / s
            writer.add_scalar_to_wandb(
                "throughput/samples_per_sec", samples_per_sec, iteration
            )
            writer.add_scalar_to_wandb(
                "throughput/samples_per_sec_per_replica",
                samples_per_sec_per_replica,
                iteration,
            )
            writer.add_scalar_to_wandb(
                "throughput/tokens_per_sec", tokens_per_sec, iteration
            )
            writer.add_scalar_to_wandb(
                "throughput/tokens_per_sec_per_replica",
                tokens_per_sec_per_replica,
                iteration,
            )
            writer.add_scalar_to_wandb(
                "throughput/tokens_per_gpu_per_sec",
                tokens_per_gpu_per_second,
                iteration,
            )
            writer.add_scalar_to_wandb(
                "throughput/tokens_per_gpu_per_sec_per_replica",
                tokens_per_gpu_per_second_per_replica,
                iteration,
            )
            writer.add_scalar_to_wandb("throughput/tflops", tflops, iteration)
            writer.add_scalar_to_wandb(
                "throughput/approx_params_in_billions",
                approx_parameters_in_billions,
                iteration,
            )
            writer.add_scalar_to_wandb(
                "throughput/elapsed_ms_per_iteration",
                elapsed_time_per_iteration,
                iteration,
            )
            if loss_dict is not None:
                for k, v in loss_dict.items():
                    writer.add_scalar_to_wandb(f"loss/{k}", v, iteration)

            if args.log_timers_to_tensorboard:
                writer.add_scalar(
                    "iteration-time/iteration-time",
                    elapsed_time_per_iteration,
                    iteration,
                )
                writer.add_scalar(
                    "iteration-time/iteration-time vs samples",
                    elapsed_time_per_iteration,
                    args.consumed_train_samples,
                    x_axis_samples,
                )
                writer.add_scalar(
                    "iteration-time/iteration-time vs tokens",
                    elapsed_time_per_iteration,
                    args.consumed_train_tokens,
                    x_axis_tokens,
                )
        log_string = " iteration {:8d}/{:8d} |".format(iteration, args.train_iters)
        log_string += " consumed samples: {:12d} |".format(args.consumed_train_samples)
        log_string += " consumed tokens: {:12d} |".format(args.consumed_train_tokens)
        log_string += " elapsed time per iteration (ms): {:.1f} |".format(
            elapsed_time_per_iteration * 1000.0
        )
        log_string += " learning rate: {:.3E} |".format(learning_rate)
        log_string += " global batch size: {:5d} |".format(batch_size)

        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                avg = total_loss_dict[key].item() / float(
                    max(1, total_loss_dict[advanced_iters_key])
                )
                if avg > 0.0:
                    log_string += " {}: {:.6E} |".format(key, avg)
                # total_loss_dict[key] = get_accelerator().FloatTensor([0.0])
                total_loss_dict[key] = torch.tensor([0.0]).to(DEVICE)
        if loss_scale is not None:
            log_string += " loss scale: {:.1f} |".format(loss_scale)
        if grad_norm is not None:
            log_string += " grad norm: {:.3f} |".format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += " num zeros: {:.1f} |".format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += " params norm: {:.3f} |".format(params_norm)
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            log_string += " curriculum seqlen: {:5d} |".format(args.curriculum_seqlen)
        if args.random_ltd:
            log_string += " random ltd reserved length: {:5d} |".format(
                args.random_ltd_reserved_length
            )
        log_string += " actual seqlen: {:5d} |".format(seq_len)
        log_string += " number of skipped iterations: {:3d} |".format(
            total_loss_dict[skipped_iters_key]
        )
        log_string += " number of nan iterations: {:3d} |".format(
            total_loss_dict[nan_iters_key]
        )
        log_string += " samples per second: {:.3f} |".format(samples_per_sec)
        log_string += " tokens per gpu per second (tgs): {:.3f} |".format(
            tokens_per_gpu_per_second
        )
        log_string += " TFLOPs: {:.2f} |".format(tflops)
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        # print_rank_last(log_string)
        log.info(log_string)
        if report_memory_flag and learning_rate > 0.0:
            # Report memory after optimizer state has been initialized.
            report_memory("(after {} iterations)".format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag
