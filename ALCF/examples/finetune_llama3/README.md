# Finetune Llama3 from Hugging Face Checkpoint

1. Download HF Checkpoint:

    ```bash
    MODEL="Llama-3.2-1B"
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "meta-llama/${MODEL}" --local-dir "${MODEL}"
    ```

1. Convert HF --> MDS:

    ```bash
    git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
    cd Megatron-DeepSpeed
    curl https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json -o alpaca_data.json
    TP=1 PP=1 ZERO_STAGE=1 MODEL_NAME=Llama-3.2-1B bash ALCF/examples/finetune_llama3/finetune_llama.sh convert_hf2mds
    ```

## Dataset

You can access the dataset from [here](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json).

### Usage

#### 1. Converting Hugging Face Model Weights to Megatron-Deepspeed Model

```bash
bash examples_deepspeed/finetune_hf_llama/finetune_llama.sh convert_hf2mds
```

This command writes the Hugging Face model weights into the Megatron-Deepspeed model and saves it. You can adjust the parallel configuration in the script.```convert_mds2hf``` can convert a Megatron-Deepspeed model into the Hugging Face format

#### 2. Fine-tuning Process

```bash
bash examples_deepspeed/finetune_hf_llama/finetune_llama.sh
```

Execute this command to initiate the finetuning process. The task originates from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca.git).
