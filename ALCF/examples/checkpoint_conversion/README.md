# Converting `AutoModel` to DeepSpeed ZeRO Checkpoint

We would like to convert an (arbitrarily large) HuggingFace model to a ZeRO
checkpoint so that we can use it for continual pre-training with
Megatron-DeepSpeed.

Previously, we had been using the approach from [ALCF / examples / finetune_llama3](/ALCF/examples/finetune_llama3/README.md)




|       Model Name       | Model Size | Model Parameters | Largest Layer Parameters | Memory Needed |
|:----------------------:|:----------:|:----------------:|:------------------------:|:-------------:|
| Llama-3.3-70B-Instruct |     70B    |      69503M      |           1050M          |    70.45GB   | 


```bash
launch python3 \
  ALCF/examples/checkpoint_conversion/hf_to_zero.py \
  --zero-stage=3 \
  --device=cpu \
  --model='meta-llama/llama-3.3-70b-instruct'
```





## Estimate Memory Needs for Llama-3.3-70B-Instruct

```bash
$ python3 -c 'from transformers import AutoModel; \
∙ from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
∙ model = AutoModel.from_pretrained("meta-llama/Llama-3.3-70B-Instruct"); \
∙ estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=12, num_nodes=4)'
```

<details closed><summary>Output</summary>


```bash
Loading checkpoint shards: 100%|████████████████| 30/30 [08:28<00:00, 16.94s/it]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 4 nodes, 12 GPUs per node.
SW: Model with 69503M total params, 1050M largest layer params.
  per CPU  |  per GPU |   Options
  436.93GB |   3.91GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
 4660.54GB |   3.91GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
  388.38GB |   6.61GB | offload_param=none, offload_optimizer=cpu , zero_init=1
 4660.54GB |   6.61GB | offload_param=none, offload_optimizer=cpu , zero_init=0
   70.45GB |  28.19GB | offload_param=none, offload_optimizer=none, zero_init=1
 4660.54GB |  28.19GB | offload_param=none, offload_optimizer=none, zero_init=0
took: 0h:08m:44s
```

</details>


- Model States and Memory Needs for Llama-3.3-70B-Instruct:


    |  per CPU  | per GPU |                         Options                         |
    |:---------:|:-------:|:-------------------------------------------------------:|
    |  436.93GB |  3.91GB |  offload_param=cpu, offload_optimizer=cpu, zero_init=1  |
    | 4660.54GB |  3.91GB |  offload_param=cpu, offload_optimizer=cpu, zero_init=0  |
    |  388.38GB |  6.61GB |  offload_param=none, offload_optimizer=cpu, zero_init=1 |
    | 4660.54GB |  6.61GB |  offload_param=none, offload_optimizer=cpu, zero_init=0 |
    |  70.45GB  | 28.19GB | offload_param=none, offload_optimizer=none, zero_init=1 |
    | 4660.54GB | 28.19GB | offload_param=none, offload_optimizer=none, zero_init=0 |



