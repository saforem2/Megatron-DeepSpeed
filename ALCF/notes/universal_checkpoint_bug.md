# 🐛 Universal Checkpoint Conversion Bug (with `DP=768`)

## Table of Contents


1. [📓 Summary](#-summary)
1. [🚧 Issue](#-issue)
   1. [🔍 Running with Debugger](#-running-with-debugger)
   1. [🤔 Why is this Happening?](#-why-is-this-happening)
   1. [🧰 Proposed Fix](#-proposed-fix)
   1. [✅ Confirm Fix Works](#-confirm-fix-works)
 1. [👻 Bug Doesn't Appear for Smaller Checkpoints](#-bug-doesnt-appear-for-smaller-checkpoints)


## 📓 Summary

- ✅ Everything works _as is_ for checkpoints created on small scales (small DP ?)
  - Explicitly confrm this
    (see [👻 Bug Doesn't appear for smaller checkpoints](#-bug-doesnt-appear-for-smaller-checkpoints))
    by:

    1. Generate checkpoint from using 4 nodes of Aurora
    2. Convert this checkpoint to universal without issue

- ❌ Trying to repeat this exact same process, but using a checkpoint created
  with `DP=768` (12 nodes of Aurora) fails with a `RuntimeError`.

  We then walk through:
  - [🚧 A Description of bug](#-issue)
  - [🔍 Running with Debugger](#-running-with-debugger)
  - [🤔 Why is this Happening?](#-why-is-this-happening)
  - [🧰 Proposed fix](#-proposed-fix)
  - [✅ Confirmation of Fix](#-confirm-fix-works)


## 🚧 Issue

Trying to convert a checkpoint[^parallel-config] created with `DP=768` (12 nodes of Aurora) to
universal checkpoint format, I encountered the following `RuntimeError`:

```python
RuntimeError: narrow(): length must be non-negative.
```

[^parallel-config]: In the example described here,
    we the checkpoint was created with:

    ```yaml
    data_parallel_size: 768
    ds_sequence_parallel_size: 1
    no_pipeline_parallel: true
    pipeline_model_parallel_size: 1
    pipeline_model_parallel_split_rank: null
    sequence_parallel: false
    tensor_model_parallel_size: 1
    transformer_pipeline_model_parallel_size: 1
    ```

The full command and traceback are included below:

```bash
#[01:06:34 AM][x4705c5s4b0n0][/f/A/f/p/a/t/2/Megatron-DeepSpeed][🌱 docs-ucp-bug][?]
$ cd deps/DeepSpeed && git status && cd - && ckpt_dir=checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05 ; gs=$(cat "${ckpt_dir}/latest_checkpointed_iteration.txt") && echo "global step: ${gs}" && python3 deps/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py --input_folder "${ckpt_dir}/global_step${gs}" --output_folder "${ckpt_dir}/global_step${gs}_universal" --keep_temp_folder
```

Output:

```bash
On branch master
Your branch is up to date with 'origin/master'.

nothing to commit, working tree clean
/flare/Aurora_deployment/foremans/projects/argonne-lcf/tmp/2024-12-28-154515/Megatron-DeepSpeed
global step: 95600
[2024-12-29 01:06:45,089] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-12-29 01:06:45,434] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to xpu (auto detect)
args = Namespace(input_folder='checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05/global_step95600', output_folder='checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05/global_step95600_universal', num_extract_workers=4, num_merge_workers=2, keep_temp_folder=True, strict=True, inject_missing_state=False)
Convert DeepSpeed Checkpoint to Universal Checkpoint
Converting DeepSpeed checkpoint in checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05/global_step95600 to Universal checkpoint in checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05/global_step95600_universal
*** 1. Extracting ZeRO fragments
100%|████████████████████████▋| 767/768 [01:49<00:00,  7.01it/s]
```

before crashing with the following traceback:

```python
concurrent.futures.process._RemoteTraceback:
Traceback (most recent call last):
  File "/opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/concurrent/futures/process.py", line 246, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
  File "/lus/flare/projects/Aurora_deployment/foremans/projects/argonne-lcf/tmp/2024-12-28-154515/Megatron-DeepSpeed/deps/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py", line 114, in extract_zero_shards
    sd = ds_checkpoint.get_zero_checkpoint_state(pp_index=pp_index, tp_index=tp_index, dp_index=dp_index)
  File "/lus/flare/projects/Aurora_deployment/foremans/projects/argonne-lcf/tmp/2024-12-28-154515/Megatron-DeepSpeed/deps/DeepSpeed/deepspeed/checkpoint/deepspeed_checkpoint.py", line 124, in get_zero_checkpoint_state
    return self.zero_checkpoint.get_state_for_rank(pp_index=pp_index,
  File "/lus/flare/projects/Aurora_deployment/foremans/projects/argonne-lcf/tmp/2024-12-28-154515/Megatron-DeepSpeed/deps/DeepSpeed/deepspeed/checkpoint/zero_checkpoint.py", line 62, in get_state_for_rank
    self._strip_tensor_paddings(sd)
  File "/lus/flare/projects/Aurora_deployment/foremans/projects/argonne-lcf/tmp/2024-12-28-154515/Megatron-DeepSpeed/deps/DeepSpeed/deepspeed/checkpoint/zero_checkpoint.py", line 110, in _strip_tensor_paddings
    group_state[state_name] = torch.narrow(state_value, 0, 0, raw_length).clone()
RuntimeError: narrow(): length must be non-negative.
```

<!--
Even more interesting, this only seems to happen for checkpoints created using
more than 18 nodes of Aurora (which would correspond to a `data_parallel_size = 216`)
-->

### 🔍 Running with Debugger

Running with:

```bash
python3 -m pudb deps/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py \
  --input_folder "${ckpt_dir}/global_step${gs}" \
  --output_folder "${ckpt_dir}/global_step${gs}_universal" \
  --keep_temp_folder \
  --num_extract_workers 1 \
  --num_merge_workers 1
```

<details closed><summary>Traceback</summary>

```python
Traceback (most recent call last):
  File "/lus/flare/projects/Aurora_deployment/foremans/projects/argonne-lcf/tmp/2024-12-28-154515/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/pudb/__init__.py", line 176, in _runscript
    dbg._runscript(mainpyfile)
  File "/lus/flare/projects/Aurora_deployment/foremans/projects/argonne-lcf/tmp/2024-12-28-154515/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/pudb/debugger.py", line 529, in _runscript
    self.run(statement)
  File "/opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/bdb.py", line 598, in run
    exec(cmd, globals, locals)
  File "<string>", line 1, in <module>
  File "deps/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py", line 549, in <module>
    main(args)
  File "deps/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py", line 499, in main
    _extract_zero_shard_files(args, ds_checkpoint, temp_dir)
  File "deps/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py", line 370, in _extract_zero_shard_files
    _do_parallel_work(do_work, _3d_range_list, args.num_extract_workers)
  File "deps/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py", line 359, in _do_parallel_work
    results.append(do_work(work))
  File "deps/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py", line 114, in extract_zero_shards
    sd = ds_checkpoint.get_zero_checkpoint_state(pp_index=pp_index, tp_index=tp_index, dp_index=dp_index)
  File "/lus/flare/projects/Aurora_deployment/foremans/projects/argonne-lcf/tmp/2024-12-28-154515/Megatron-DeepSpeed/deps/DeepSpeed/deepspeed/checkpoint/deepspeed_checkpoint.py", line 124, in get_zero_checkpoint_state
    return self.zero_checkpoint.get_state_for_rank(pp_index=pp_index,
  File "/lus/flare/projects/Aurora_deployment/foremans/projects/argonne-lcf/tmp/2024-12-28-154515/Megatron-DeepSpeed/deps/DeepSpeed/deepspeed/checkpoint/zero_checkpoint.py", line 62, in get_state_for_rank
    self._strip_tensor_paddings(sd)
  File "/lus/flare/projects/Aurora_deployment/foremans/projects/argonne-lcf/tmp/2024-12-28-154515/Megatron-DeepSpeed/deps/DeepSpeed/deepspeed/checkpoint/zero_checkpoint.py", line 110, in _strip_tensor_paddings
    group_state[state_name] = torch.narrow(state_value, 0, 0, raw_length).clone()
RuntimeError: narrow(): length must be non-negative.
```

</details>

Stepping through this command:

```python
>>> group_state[state_name] = torch.narrow(state_value, 0, 0, raw_length).clone()
Traceback (most recent call last):
  File "<pudb command line>", line 1, in <module>
RuntimeError: narrow(): length must be non-negative.

>>> raw_length
-676
>>> state_value.numel()
348
>>> group_paddings[key]
1024
>>> key
1
>>> state_name
'exp_avg'
>>> key
1
```

### 🤔 Why is this Happening?

The problematic line occurs here in [deepspeed / checkpoint / `ds_to_universal.py#L114`](https://github.com/microsoft/DeepSpeed/blob/cc03c76d57f41752d8cfb84c2e45b8e0da8083da/deepspeed/checkpoint/ds_to_universal.py#L114), shown below:

```python
sd = ds_checkpoint.get_zero_checkpoint_state(
    pp_index=pp_index, tp_index=tp_index, dp_index=dp_index
)
```

The `extract_zero_shards` function tries calling the `DeepSpeedCheckpoint.get_zero_checkpoint_state` method
here [deepspeed / `checkpoint.py#L123-127`](https://github.com/microsoft/DeepSpeed/blob/cc03c76d57f41752d8cfb84c2e45b8e0da8083da/deepspeed/checkpoint/deepspeed_checkpoint.py#L123-L127), which looks like:

```python
def get_zero_checkpoint_state(self, pp_index, tp_index, dp_index) -> dict:
        return self.zero_checkpoint.get_state_for_rank(pp_index=pp_index,
                                                       tp_index=tp_index,
                                                       dp_index=dp_index,
                                                       keys_to_ignore=[PARAM_SHAPES])
```

This (^) then calls the `ZeROCheckpoint.get_state_for_rank` method from
[deepspeed / checkpoint / `zero_checkpoint.py#L53-73`](https://github.com/microsoft/DeepSpeed/blob/cc03c76d57f41752d8cfb84c2e45b8e0da8083da/deepspeed/checkpoint/zero_checkpoint.py#L53-L73).

Now, this `get_state_for_rank` function accepts an argument
`strip_tensor_paddings` which is true by default.

Calling the `strip_tensor_paddings` method (with `strip_tensor_paddings=True`), we hit:

```python
for state_file in state_file_list:
    # ...clipped...
    if strip_tensor_paddings:
        self._strip_tensor_paddings(sd)  # <-- this is where the error is raised
```

and we hit the `RuntimeError` when calling this
`self._strip_tensor_paddings(sd)` method.

Stepping into the `self._strip_tensor_paddings` method, we see that it calls
[this block](https://github.com/microsoft/DeepSpeed/blob/cc03c76d57f41752d8cfb84c2e45b8e0da8083da/deepspeed/checkpoint/zero_checkpoint.py#L108-L110)
(shown below) which calculates the `raw_length` as:

```python
def _strip_tensor_paddings(self, sd):
    param_group_states = self._get_param_group_states(sd)
    if param_group_states is None:
        return

    group_paddings = self._get_optimizer_state(sd, GROUP_PADDINGS)
    if group_paddings is None:
        return

    for key, group_state in param_group_states.items():
        if group_paddings[key] == 0:
            continue
        for state_name, state_value in group_state.items():
            if state_name != "step" and torch.is_tensor(state_value):
                # 🐛 see debugger output below
                raw_length = state_value.numel() - group_paddings[key]  # <-- this is negative
                group_state[state_name] = torch.narrow(state_value, 0, 0, raw_length).clone()
            else:
                group_state[state_name] = state_value
```

which, when `raw_length` is negative, causes:

```python
group_state[state_name] = torch.narrow(state_value, 0, 0, raw_length).clone()
RuntimeError: narrow(): length must be non-negative.
```

It wasn't immediately obvious what this `strip_tensor_paddings` argument represents
(or even what the method is doing, to be honest), so I didn't have much insight 
into why this would only be happening for checkpoints created at larger scales.


### 🧰 Proposed Fix

Naively, the first (and easiest) thing to try was to see if I could just skip this
`strip_tensor_paddings` step by setting `strip_tensor_paddings=False` in the
call to (1) `get_zero_checkpoint_state` in the (2) `extract_zero_shards` function.

Unfortunately, since (1) `DeepSpeedCheckpoint.get_zero_checkpoint_state()`
**DOES NOT** take in the `strip_tensor_paddings` argument,
there is no way to pass this along to the (2) `ZeROCheckpoint.get_state_for_rank()` call.

So, our proposed fix requires two modifications:

1. Modify `DeepSpeedCheckpoint.get_zero_checkpoint_state` signature from [here](https://github.com/microsoft/DeepSpeed/blob/cc03c76d57f41752d8cfb84c2e45b8e0da8083da/deepspeed/checkpoint/deepspeed_checkpoint.py#L123) to accept the `strip_tensor_paddings` argument:

    ```diff
    warning: Empty last update token.
    diff --git a/deepspeed/checkpoint/deepspeed_checkpoint.py b/deepspeed/checkpoint/deepspeed_checkpoint.py
    index 31997177..a2ef5d0d 100644
    --- a/deepspeed/checkpoint/deepspeed_checkpoint.py
    +++ b/deepspeed/checkpoint/deepspeed_checkpoint.py
    @@ -120,11 +120,12 @@ class DeepSpeedCheckpoint(object):
             self.global_state[ITERATION_KEY] = sd.get(ITERATION_KEY, 0)
             self.global_state[ARGS_KEY] = sd.get(ARGS_KEY, None)
    
    -    def get_zero_checkpoint_state(self, pp_index, tp_index, dp_index) -> dict:
    -        return self.zero_checkpoint.get_state_for_rank(pp_index=pp_index,
    +    def get_zero_checkpoint_state(self, pp_index, tp_index, dp_index, strip_tensor_paddings: bool = True) -> dict:
    +        return self.zero_checkpoint.get_state_for_rank(pp_index=pp_index,  # type:ignore
                                                            tp_index=tp_index,
                                                            dp_index=dp_index,
    -                                                       keys_to_ignore=[PARAM_SHAPES])
    +                                                       keys_to_ignore=[PARAM_SHAPES],
    +                                                       strip_tensor_paddings=strip_tensor_paddings)
    
         def get_zero_files(self, pp_index, tp_index, dp_index) -> list:
             return self.zero_checkpoint.get_files_for_rank(pp_index=pp_index, tp_index=tp_index, dp_index=dp_index)
    ```

1. With this in place, we can now try setting `strip_tensor_paddings = False` in the call shown below:

    ```diff
    diff --git a/deepspeed/checkpoint/ds_to_universal.py b/deepspeed/checkpoint/ds_to_universal.py
    index f7b75eee..cbbbef6b 100755
    --- a/deepspeed/checkpoint/ds_to_universal.py
    +++ b/deepspeed/checkpoint/ds_to_universal.py
    @@ -111,7 +111,7 @@ def _save_checkpoint(file_path, chkpt_sd):
     
     def extract_zero_shards(dir, ds_checkpoint, indices_3D):
         pp_index, tp_index, dp_index = indices_3D
    -    sd = ds_checkpoint.get_zero_checkpoint_state(pp_index=pp_index, tp_index=tp_index, dp_index=dp_index)
    +    sd = ds_checkpoint.get_zero_checkpoint_state(pp_index=pp_index, tp_index=tp_index, dp_index=dp_index, strip_tensor_paddings=False)
    ```

### ✅ Confirm Fix Works

We've added the proposed changes above to the `saforem2/ucp-bug` branch.

We can confirm explicitly that the proposed fix works by retrying the conversion:

```bash
$ cd deps/DeepSpeed && git status && git checkout 'saforem2/ucp-bug' && PAGER='' git diff deepspeed/checkpoint/ && cd - && ckpt_dir=checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05 ; gs=$(cat "${ckpt_dir}/latest_checkpointed_iteration.txt") && echo "global step: ${gs}" && python3 deps/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py --input_folder "${ckpt_dir}/global_step${gs}" --output_folder "${ckpt_dir}/global_step${gs}_universal" --keep_temp_folder
On branch saforem2/ucp-bug
nothing to commit, working tree clean
Already on 'saforem2/ucp-bug'
/flare/Aurora_deployment/foremans/projects/argonne-lcf/tmp/2024-12-28-154515/Megatron-DeepSpeed
global step: 95600
[2024-12-29 01:58:23,658] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-12-29 01:58:30,635] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to xpu (auto detect)
args = Namespace(input_folder='checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05/global_step95600', output_folder='checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05/global_step95600_universal', num_extract_workers=4, num_merge_workers=2, keep_temp_folder=True, strict=True, inject_missing_state=False)
Convert DeepSpeed Checkpoint to Universal Checkpoint
Converting DeepSpeed checkpoint in checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05/global_step95600 to Universal checkpoint in checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05/global_step95600_universal
*** 1. Extracting ZeRO fragments
100%|██████████████████████████| 768/768 [04:15<00:00,  3.00it/s]
*** 2. Merging slices .....
100%|██████████████████████████| 195/195 [01:11<00:00,  2.74it/s]
*** 3. Saving common optimizer states
*** Done!
took: 0h:09m:00s
```

fixed!

## 👻 Bug Doesn't Appear for Smaller Checkpoints

As a sanity check, we can explicitly test that everything works
_as is_ when converting smaller checkpoints to universal format.

1. Create checkpoint on 4 nodes of Aurora

    ```bash
    $ PBS_O_WORKDIR=$(pwd) DATA_FILE_LIST=ALCF/data-lists/aurora/books.txt NLAYERS=1 SAVE_INTERVAL=10 bash train_aGPT_7B.sh
    # ...irrelevant output clipped...
    [2024-12-29 00:41:41.185272][INFO][utils.py:368] successfully saved checkpoint at iteration 690 to checkpoints/ws48_ds_stage1_nl1_hs4096_mb1_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr_lwf_flash
    ```

2. Convert to universal checkpoint using `DeepSpeed` master (**unchanged**), and
   confirm that it works without issue.

Using the checkpoint generated in 1., we use
[DeepSpeed / deepspeed / checkpoints / `ds_to_universal.py`](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/checkpoint/ds_to_universal.py)
**as is**:

```bash
#[🐍 aurora_nre_models_frameworks-2024.2.1_u1](👻 aurora_nre_models_frameworks-2024.2.1_u1)
#[12:46:03 AM][x4705c5s4b0n0][/f/A/f/p/a/t/2/Megatron-DeepSpeed][🌱 docs-ucp-bug][?]
$ cd deps/DeepSpeed && git status && cd - && ckpt_dir=checkpoints/ws48_ds_stage1_nl1_hs4096_mb1_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr_lwf_flash ; gs=$(cat "${ckpt_dir}/latest_checkpointed_iteration.txt") && echo "global step: ${gs}" && python3 deps/DeepSpeed/deepspeed/checkpoint/ds_to_universal.py --input_folder "${ckpt_dir}/global_step${gs}" --output_folder "${ckpt_dir}/global_step${gs}_universal" --keep_temp_folder
On branch master # on master
Your branch is up to date with 'origin/master'.  # no local changes
nothing to commit, working tree clean
/flare/Aurora_deployment/foremans/projects/argonne-lcf/tmp/2024-12-28-154515/Megatron-DeepSpeed
global step: 690  # <-- ckpt from 1.
[2024-12-29 00:46:14,466] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-12-29 00:46:14,832] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to xpu (auto detect)
args = Namespace(input_folder='checkpoints/ws48_ds_stage1_nl1_hs4096_mb1_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr_lwf_flash/global_step690', output_folder='checkpoints/ws48_ds_stage1_nl1_hs4096_mb1_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr_lwf_flash/global_step690_universal', num_extract_workers=4, num_merge_workers=2, keep_temp_folder=True, strict=True, inject_missing_state=False)
Convert DeepSpeed Checkpoint to Universal Checkpoint
Converting DeepSpeed checkpoint in checkpoints/ws48_ds_stage1_nl1_hs4096_mb1_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr_lwf_flash/global_step690 to Universal checkpoint in checkpoints/ws48_ds_stage1_nl1_hs4096_mb1_seq4096_gb768_sp1_pp1_tp1_bf16_optadamw_lr_lwf_flash/global_step690_universal
*** 1. Extracting ZeRO fragments
100%|██████████████████████████████████| 48/48 [00:19<00:00,  2.44it/s]
*** 2. Merging slices .....
100%|██████████████████████████████████| 9/9 [00:05<00:00,  1.68it/s]
*** 3. Saving common optimizer states
*** Done!
took: 0h:01m:40s
```
