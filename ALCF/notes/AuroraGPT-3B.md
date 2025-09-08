# AuroraGPT-3B-v1

## Organization

```
/flare/AuroraGPT/AuroraGPT-v1/3B/Megatron-DeepSpeed/
```

## Llama-3.2-3B Base Config


- Model Architecture:
  - <details closed><summary>model architecture (yaml)</summary>

    ```yaml
    DP: 1536
    FFN_HIDDEN_SIZE: 8192
    GAS:
      - 2
      - 4
    GBS: 12288
    HIDDEN: 3072
    HEADS: 32
    MBS: 4
    NLAYERS: 28
    NUM_KV_HEAD: 8
    OPT: ipex.fusedlamb
    PP: 1
    SEQ: 8192
    SP: 1
    TP: 1
    USE_ACTIVATION_CHECKPOINTING: 1
    ZERO_STAGE: 1
    ```

  </details>

- 128 Nodes:
  - [pleasant-snowflake-3180](https://wandb.ai/aurora_gpt/AuroraGPT/runs/y9r7r3mh)

    ```bash
    GRAD_ACC_STEPS=2 \
        MICRO_BATCH=4 \
        USE_ACTIVATION_CHECKPOINTING=1 \
        ZERO_STAGE=0 \
        MODEL_ARCH=3B \
        OPT=ipex.fusedlamb \
        bash train_alcf.sh
    ```

- 64 Nodes:
  - [jolly-silence-3185](https://wandb.ai/aurora_gpt/AuroraGPT/runs/28qxlycg)

    ```bash
    GRAD_ACC_STEPS=4 \
        MICRO_BATCH=4 \
        USE_ACTIVATION_CHECKPOINTING=1 \
        ZERO_STAGE=0 \
        MODEL_ARCH=3B \
        OPT=ipex.fusedlamb \
        bash train_alcf.sh
    ```


## Reference Configs

- [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B):
  - [meta-llama/Llama-3.2-3B/config.json](https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json)
- [Gemma-3](https://huggingface.co/google/gemma-3-4b-it):
  - [google/gemma-3-4b-it/config.json](https://huggingface.co/google/gemma-3-4b-it/blob/main/config.json)
- [SmolLM3-3B-Base](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base):
  - [HuggingFaceTB/SmolLM3-3B-Base/config.json](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base/blob/main/config.json)


## Experiments

Would like to run:

- Large Batch Training
  - Identify largest number of tokens / batch that preserves stability
- Optimizer(s):
  - Muon
  - Lamb
  - AdamW
    - schedulefree (?)
  - Others?
- Tokenizer(s):
  - Implementation:
    - Llama Tokenizer
    - Gemma Tokenizer
  - GitHub Repos:
    - [M4THYOU/TokenDagger](https://huggingface.co/M4THYOU/TokenDagger)
    - [openai/TikToken](https://github.com/openai/tiktoken)
    - [google/sentencepiece](https://github.com/google/sentencepiece)
