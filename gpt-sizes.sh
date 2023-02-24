###############################################################################
### Main configs
## GPT-3 models use 2K sequence length/context window

## The "GPT-3 XXX" below are configs from GPT-3 paper
## https://arxiv.org/abs/2005.14165, choose based on
## your desired model size or build your own configs

## init_std is standard deviation for weight initialization. Usually larger
## model needs lower std. We used a heuristic equation of sqrt(1/3/hidden_size)
## from the MT-NLG 530B work (https://arxiv.org/pdf/2201.11990.pdf)
#
## GPT-3 Small 125M
# model_size=0.125
# num_layers=12
# hidden_size=768
# num_attn_heads=12
# global_batch_size=256
# lr=6.0e-4
# min_lr=1.0e-6
# init_std=0.02

## GPT-3 Medium 350M
# model_size=0.35
# num_layers=24
# hidden_size=1024
# num_attn_heads=16
# global_batch_size=256
# lr=3.0e-4
# min_lr=1.0e-6
# init_std=0.018

## GPT-3 Large 760M
# model_size=0.76
# num_layers=24
# hidden_size=1536
# num_attn_heads=16
# global_batch_size=256
# lr=2.5e-4
# min_lr=1.0e-6
# init_std=0.015

## GPT-3 XL 1.3B
# model_size=1.3
# num_layers=24
# hidden_size=2048
# num_attn_heads=16
# global_batch_size=512
# # lr=2.0e-4
# lr=$1
# min_lr=1.0e-6
# init_std=0.013

## GPT-3 2.7B
# MODEL_SIZE="2.7B"
# NLAYERS=32
# HIDDEN=2560
# ATEN_HEADS=32
# GLOBAL_BATCH=512
# model_size=2.7
# num_layers=32
# hidden_size=2560
# num_attn_heads=32
# global_batch_size=512

## GPT-3 6.7B
# MODEL_SIZE="6.7B"
# NLAYERS=32
# HIDDEN=4096
# ATEN_HEADS=32
# GLOBAL_BATCH=1024
# model_size=6.7
# num_layers=32
# hidden_size=4096
# num_attn_heads=32
# global_batch_size=1024

## GPT-3 13B
# model_size=13
# num_layers=40
# hidden_size=5120
# num_attn_heads=40
# global_batch_size=1024

## GPT-3 175B
# model_size=175
# num_layers=96
# hidden_size=12288
# num_attn_heads=96
# global_batch_size=1536

