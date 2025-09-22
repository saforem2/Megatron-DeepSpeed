# Megatron-DeepSpeed, optimizers, hyperparameters and CPT

Single command to test and run Megatron-DeepSpeed:

```bash
now=$(date +'%Y-%m-%d-%H%M%S') && debug_dir="${now}" && mkdir -p "${debug_dir}"&& cd "${debug_dir}"&& git clone https://github.com/argonne-lcf/Megatron-DeepSpeed && cd Megatron-DeepSpeed && source <(curl -L https://bit.ly/ezpz-utils) && ezpz_setup_env && python3 -m pip install --require-virtualenv "git+https://github.com/saforem2/ezpz" "numpy<2" deepspeed tensorboard && ezpz-test && DATA_FILE_LIST=ALCF/data-lists/aurora/books.txt bash train_alcf.sh
```
## Optimizers
The default optimizer is `adamw`. Go to this [list of optimizers](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/arguments.py#L1485) for a complete list of supported optimizers (note that dshampoo might throw checkpointing errors, we are working on fixing this). For example, to run with `muon`, you can do:
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh
```
Here
```bash
DATA_FILE_LIST=path/to/your/tokenized/data
TRAIN_TOKENS= number of training tokens
GRAD_ACC_STEPS=number of grad accumulation steps
LR=learning rate
LR_WARMUP_FRACTION=warmup fraction
OPT=optimizer
```
Your global batch size will be: `num_gpus*micro_batch_size*GRAD_ACC_STEPS`, micro batch size is 1 by default, you can change it by adding `MICRO_BATCH=new_micro_batch_size` to your options. To have the corresponding number if tokens per step, you need to multiply the global batch size by the sequence length (set with `SEQ_LEN`, default is 4096)

### Adding custom optimizers
To add a custom optimizer, you have to modify the following files:
- `megatron/optimizer/__init__.py`: [muon example](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/optimizer/__init__.py#L434), note that you either heve to import the optimizer from a pre-installed package or add it in the `megatron/optimizer/` folder.
- `megatron/arguments.py`: [optimizer arguments](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/arguments.py#L1070), to add the optimizer arguments
- `megatron/arguments.py`: [list of valid optimizers](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/arguments.py#L1485), to add the new optimizer to the list of valid optimizers

### Schedulers
Note that the default scheduler is `cosine`. We also support `infinite cosine, infinite inverse square root, constant, constant with cooldown, inverse square root, linear` schedulers. For example to change the scheduler to `constant`, you can do so with the `LR_DECAY_STYLE` option:
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR_DECAY_STYLE=constant LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh
```
To add cooldown, you need to add the `--lr_constant_plus_cooldown` flag and set the cooldown fraction with `--lr_constant_plus_cooldown_frac`. The default cooldown fraction is 0.05
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR_DECAY_STYLE=constant LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh --lr_constant_plus_cooldown --lr_constant_plus_cooldown_frac 0.01
```
#### Adding custom schedulers
To add a custom scheduler, you have to modify the following files:
- `megatron/optimizer_param_scheduler.py`: [schedulers](megatron/optimizer_param_scheduler.py), to add the new scheduler
- `megatron/arguments.py`: [list of LR arguments](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/arguments.py#L1671), to add the new scheduler arguments.
- You might have to change [the function](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/994f2a129d465cc50e6c35af075eb3292874effe/megatron/training.py#L559) to incorporate your custom scheduler options.

## Hyperparameter tuning
#### Init variance
Weight initialization is key to training LLMs,and to avoid spikes in losses. Here, we initialize the wights following this [paper](https://arxiv.org/pdf/2312.16903). The default variance value at initialization is 0.02. To add custom variances, one can use `--init-method-std, `--adjust-word-embedding-init`, and `--word-embedding-init-std`. For our runs, we do
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR_DECAY_STYLE=constant LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh --lr_constant_plus_cooldown --init-method-std ${sqrt(2/5d)}  --adjust-word-embedding-init --word-embedding-init-std 0.632
```
where `d=hidden size`.

### Learning rate
For the learning rate, we implemented the learning rate finder routine [here](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html) and [here](https://arxiv.org/pdf/1506.01186). This is activated with the `--lr-finder` and run for `TRAIN_ITERS` steps. For example, for a 1000 steps:
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/books.txt TRAIN_ITERS=1000 GRAD_ACC_STEPS=16 LR_DECAY_STYLE=constant LR=0.0002 LR_WARMUP_FRACTION=0.01 OPT=muon bash train_alcf.sh --lr_constant_plus_cooldown --init-method-std ${sqrt(2/5d)}  --adjust-word-embedding-init --word-embedding-init-std 0.632 --lr-finder
```
### Maximal Update Parametrization/Complete Parametrization
We have MuP and CompleteP incorporated in AuroraGPT. Will add details about activating them soon




## Doing CPT
### Legacy agpt-7b checkpoints
This is for doing CPT on the initial agpt-7B checkpoint where a cosine scheduler was used from `lr=0.0002` to 0. Here, the CPT stratregy followed is the replay+rewarm one where we replay a small amount of data from the initial pretraining dataset and mix it with the cpt one. The steps are as follows:
1. First, we need to train from an universal checkpoint. If you don't have the universal checkpoint, you can follow [the instructions here]($ cd deps/DeepSpeed && git status && cd - && ckpt_dir=checkpoints/ws48_ds_stage1_nl1_hs4096_mb1_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr_lwf_flash ; gs=$(cat "${ckpt_dir}/latest_checkpointed_iteration.txt") && echo "global step: ${gs}" && python3 deps/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py --input_folder "${ckpt_dir}/global_step${gs}" --output_folder "${ckpt_dir}/global_step${gs}_universal" --keep_temp_folder
).
2. Build your cpt dataset, here we are mixing the lucid papers with weight 0.9 and dolma with weight 0.1 (you can play with the weights if needed):
```bash
python3 mix_datasets.py --input 0.9 /flare/Aurora_deployment/AuroraGPT/datasets/papers/papers.txt 0.1 /flare/Aurora_deployment/AuroraGPT/datasets/dolma/dolma_v1_7_file_list_v2.txt > ${debug_dir}/Megatron-DeepSpeed/ALCF/data-lists/aurora/mix_lucid_papers09_dolma01.txt
```
3. Then, we can run the following cpt command from the Megatron-deepspeed folder (you can modify GRAD_ACC_STEPS according to the batch size you want to do CPT with):
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/mix_lucid_papers_dolma.txt LOAD=/flare/AuroraGPT/AuroraGPT-v0/checkpoint-copies/checkpoints/ws768_ds_stage1_nl32_hs4096_mb1_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr_lwf_flash TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR=0.0002 LR_WARMUP_FRACTION=0.01 bash train_alcf.sh --universal-checkpoint --finetune
```
Here, we are rewarming to the original learning but you can rewarm to any LR you seem fit.
Here the following options options/flags should be:
```bash
DATA_FILE_LIST=path/to/your/tokenized/data
LOAD=path/to/your/universal/checkpoint
SAVE=path/to/where/you/want/to/save/checkpoints
--universal-checkpoint to load a universal checkpoint
```
### New agpt runs (phase 1 -> phase 2)
For the new runs, we are using a constant LR with cooldowns. The advantage of using a constant LR is to forego the need of rewarming. Furthermore, with the cooldown, one can train a model to convergence at any point of training without committing to a token budget.

To do CPT here, 
1. Convert checkpoint to an universal checkpoint, **YOU NEED TO USE A CHECKPOINT AT LR=LR_max BEFORE COOLING DOWN**
2.  Mix the datasets as above
3. set `export LR_WARMUP_FRAC=0.0` in order to not rewarm
4. Run
```bash
DATA_FILE_LIST=./ALCF/data-lists/aurora/mix_lucid_papers_dolma.txt LOAD=/flare/AuroraGPT/AuroraGPT-v0/checkpoint-copies/checkpoints/ws768_ds_stage1_nl32_hs4096_mb1_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr_lwf_flash TRAIN_TOKENS=$((22*10**9)) GRAD_ACC_STEPS=16 LR=0.0002 LR_DECAY_STYLE=constant bash train_alcf.sh --universal-checkpoint --finetune --lr_constant_plus_cooldown 
```
5. If loss curve not recovering after a while or loss is diverging, one can take a converged checkpoint (i.e after cooling it down) and experiment with rewarming the LR to a different value, data mixing strategy etc as we did with the legacy model.
