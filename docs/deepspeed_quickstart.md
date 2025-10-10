# Megatron-DeepSpeed Quickstart

This guide walks through setting up a small scale Megatron-DeepSpeed experiment on a single machine.  It focuses on the minimum
steps required to verify an installation and run a DeepSpeed-enabled GPT pretraining job using CPU-friendly defaults.  The
commands are designed to mirror the automated checks executed in CI so that contributors can easily reproduce them locally.

## 1. Prerequisites

* **Python** 3.10 or newer.  The GitHub Actions workflow uses the `deepspeed/gh-builder` Python 3.10 container.
* **PyTorch** and CUDA toolchains are handled through the `pip install -e .` step described below.  For CPU-only validation the
  default wheels are sufficient.
* **GPU access (optional).**  Large scale training requires NVIDIA GPUs with recent CUDA drivers, but the included quickstart
  scripts and tests can execute purely on CPU.

## 2. Install Megatron-DeepSpeed

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

The development requirements file installs both runtime and testing dependencies so that the `pytest` suites work out-of-the-box.

## 3. Download tokenizers and sample dataset (optional)

For the minimal functional test we only need the GPT-2 tokenizer.  The repository ships a helper script that downloads the
pretrained vocabulary and merge files:

```bash
bash dataset/download_vocab.sh
```

To exercise the end-to-end data pipeline you can also run:

```bash
bash dataset/download_ckpt.sh
```

The checkpoint is not required for the quick functional run, but it allows you to experiment with text generation and evaluation
scripts located in `examples/` and `tests/`.

## 4. Run the test suites

Executing the unit and transformer test collections is the fastest way to confirm the environment is set up correctly.  These are
identical to the jobs wired into the CI workflow.

```bash
pytest tests/unit_tests
pytest tests/transformer
```

Both suites are CPU friendly and complete in a few minutes on a modern workstation.  Failures usually indicate missing
dependencies or incompatible PyTorch builds.

## 5. Launch a DeepSpeed GPT sanity check

The `examples_deepspeed` directory contains launcher templates configured for DeepSpeed integration.  To run a small GPT
pretraining experiment:

1. Copy the template config: `cp examples_deepspeed/rebase/ds_config_gpt_TEMPLATE.json ds_config_gpt_local.json`.
2. Edit the copy to adjust ZeRO stage, batch sizes, and optimizer settings for your hardware.
3. Launch training with a single GPU (or CPU for functionality checks):

   ```bash
   deepspeed --num_gpus=1 pretrain_gpt.py \
     --tensor-model-parallel-size 1 \
     --pipeline-model-parallel-size 1 \
     --micro-batch-size 2 \
     --global-batch-size 16 \
     --seq-length 512 \
     --max-position-embeddings 512 \
     --num-layers 2 \
     --hidden-size 256 \
     --num-attention-heads 4 \
     --data-path /path/to/my-gpt2_text_document \
     --vocab-file dataset/gpt2-vocab.json \
     --merge-file dataset/gpt2-merges.txt \
     --save-interval 1000 \
     --deepspeed \
     --deepspeed_config ds_config_gpt_local.json
   ```

Adjust the parallel sizes when scaling beyond a single device.  When executing on CPU-only hosts, add
`--no-masked-softmax-fusion` and reduce the batch sizes to keep iteration times manageable.

## 6. Next steps

* Review [`CONTRIBUTING.md`](../CONTRIBUTING.md) for coding standards, documentation expectations, and testing policies.
* Explore the recipes under `examples_deepspeed/azureml` for production-grade Azure Machine Learning deployments.
* Consult the upstream Megatron-LM README (embedded below the repository overview) for advanced configuration details once you
  are comfortable with the DeepSpeed-specific workflow.
