# AuroraGPT-v1 (Small)

- [WandB Report: AuroraGPT-v1 (Small)](https://api.wandb.ai/links/aurora_gpt/5qxfdak3)

## 📊 Performance Results

| model-arch | `nlayers` | model size (B) | `tpgps` | `TFLOPs` |
| :--------: | :-------: | :------------: | :-----: | :------: |
|   Llama3   |     4     |      1.98      |  8018   |  66.75   |
|   Llama3   |     6     |      2.18      |  6786   |  68.76   |
|   Llama3   |     8     |      2.38      |  5874   |  70.37   |
|   Llama3   |    10     |      2.58      |  5179   |  71.36   |
|   Llama3   |    12     |      2.78      |  4646   |  72.34   |
|   Llama3   |    14     |      2.98      |  4202   |  73.15   |
|  SmolLM3   |     8     |      1.67      |  7316   |  61.59   |
|  SmolLM3   |    10     |      1.83      |  6448   |  62.72   |
|  SmolLM3   |    12     |      1.99      |  5780   |  64.09   |
|  SmolLM3   |    14     |      2.14      |  5238   |  64.82   |
|  SmolLM3   |    16     |      2.30      |  4734   |  64.77   |
|  SmolLM3   |    18     |      2.45      |  4363   |  65.38   |

- Note: OOM for LLama3 @ 16 layers and SmolLM3 @ 20 layers

## ⚙️ Configs

- Explicit command:

    ```bash
    MODEL_ARCH={smollm3-3B,llama3-3B} \
        NLAYERS=<nlayers> \
        GRAD_ACC_STEPS=2 \
        MICRO_BATCH=1 \
        USE_ACTIVATION_CHECKPOINTING=0 \
        ZERO_STAGE=0 \
        OPT=adamw \
        LR_DECAY_STYLE=constant \
        TOKENIZER_TYPE=HFTokenizer \
        TOKENIZER_MODEL=google/gemma-7b \
        DATA_FILE_LIST=ALCF/data-lists/$(ezpz_get_machine_name)/books.txt \
        bash train_alcf.sh
    ```


- Llama3 Architecture:

    ```llama3-architecture.yaml
    HEADS: 24
    HIDDEN: 3072
    FFN_HIDDEN_SIZE: 8192
    NLAYERS: XX
    NUM_KV_HEAD: 8
    SEQ: 8192
    USE_ACTIVATION_CHECKPOINTING: 0
    ZERO: 0
    ```

- SmolLM3 Architecture:

    ```smollm3-architecture.yaml
    HEADS: 16
    HIDDEN: 2048
    FFN_HIDDEN_SIZE: 11008
    NLAYERS: XX
    NUM_KV_HEAD: 4
    SEQ: 8192
    USE_ACTIVATION_CHECKPOINTING: 0
    ZERO: 0
    ```

[^lmax]: This is (~ roughly) at memory capacity

## Raw Data

- TFLOPs data:

    ```bash
    [Aurora] [NHOST:2][MB:1][GAS:2][GB:48]  [@ 2025-08-21-124405][id:Aurora, 2, llama3-3B-nLayers14, 48, 2025-08-21-124405][TFLOPS-lm:73.04185]	73.15212991672594
    [Aurora] [NHOST:2][MB:1][GAS:2][GB:48]  [@ 2025-08-21-133138][id:Aurora, 2, llama3-3B-nLayers12, 48, 2025-08-21-133138][TFLOPS-lm:72.30713]	72.34283930368927
    [Aurora] [NHOST:2][MB:1][GAS:2][GB:48]  [@ 2025-08-21-134052][id:Aurora, 2, llama3-3B-nLayers10, 48, 2025-08-21-134052][TFLOPS-lm:71.56954]	71.56953772155975
    [Aurora] [NHOST:2][MB:1][GAS:2][GB:48]  [@ 2025-08-21-122810][id:Aurora, 2, llama3-3B-nLayers10, 48, 2025-08-21-122810][TFLOPS-lm:71.26275]	71.36638292006606
    [Aurora] [NHOST:2][MB:1][GAS:2][GB:48]  [@ 2025-08-21-134526][id:Aurora, 2, llama3-3B-nLayers8, 48, 2025-08-21-134526][TFLOPS-lm:70.29466]	70.37302585667582
    [Aurora] [NHOST:2][MB:1][GAS:2][GB:48]  [@ 2025-08-21-142250][id:Aurora, 2, llama3-3B-nLayers6, 48, 2025-08-21-142250][TFLOPS-lm:68.81743]	68.75536285086024
    [Aurora] [NHOST:2][MB:1][GAS:2][GB:48]  [@ 2025-08-21-143234][id:Aurora, 2, llama3-3B-nLayers4, 48, 2025-08-21-143234][TFLOPS-lm:66.88754]	66.7547450957777

    [Aurora] [NHOST:2][MB:1][GAS:2][GB:48]  [@ 2025-08-21-131910][id:Aurora, 2, smollm3-nLayers18, 48, 2025-08-21-131910][TFLOPS-lm:66.34897]	66.319195539024
    [Aurora] [NHOST:2][MB:1][GAS:2][GB:48]  [@ 2025-08-21-125937][id:Aurora, 2, smollm3-nLayers16, 48, 2025-08-21-125937][TFLOPS-lm:65.7598]	65.87611405627682
    [Aurora] [NHOST:2][MB:1][GAS:2][GB:48]  [@ 2025-08-21-140507][id:Aurora, 2, smollm3-nLayers14, 48, 2025-08-21-140507][TFLOPS-lm:12.39282]	65.57458612915367
    [Aurora] [NHOST:2][MB:1][GAS:2][GB:48]  [@ 2025-08-21-135339][id:Aurora, 2, smollm3-nLayers12, 48, 2025-08-21-135339][TFLOPS-lm:64.87912]	64.89387696083435
    [Aurora] [NHOST:2][MB:1][GAS:2][GB:48]  [@ 2025-08-21-134749][id:Aurora, 2, smollm3-nLayers10, 48, 2025-08-21-134749][TFLOPS-lm:63.40927]	63.266400862713915
    [Aurora] [NHOST:2][MB:1][GAS:2][GB:48]  [@ 2025-08-21-135130][id:Aurora, 2, smollm3-nLayers8, 48, 2025-08-21-135130][TFLOPS-lm:62.34549]	62.28304152993416
    ```

- LLama3 tokens per gpu per second:

    ```bash
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers14, args.global_batch_size: 48, created_at: 2025-08-21-124405	4202.008365924828
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers12, args.global_batch_size: 48, created_at: 2025-08-21-133138	4646.389722691857
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers12, args.global_batch_size: 96, created_at: 2025-08-21-121138	4986.43082836443
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers10, args.global_batch_size: 48, created_at: 2025-08-21-122810	5179.048714554391
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers8, args.global_batch_size: 48, created_at: 2025-08-21-134526	5874.068043287903
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers6, args.global_batch_size: 48, created_at: 2025-08-21-142250	6786.922466326357
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers4, args.global_batch_size: 48, created_at: 2025-08-21-143234	8018.191437878496
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers10, args.global_batch_size: 48, created_at: 2025-08-21-134749	6445.155290877832
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers8, args.global_batch_size: 48, created_at: 2025-08-21-135130	7327.6685718892195
    ```

- SmolLM3 (global) tokens per second:


    ```bash
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers18, args.global_batch_size: 48, created_at: 2025-08-21-131910	104714.98091017039
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers16, args.global_batch_size: 48, created_at: 2025-08-21-125937	113606.28598021298
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers14, args.global_batch_size: 48, created_at: 2025-08-21-140507	125730.55726532736
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers12, args.global_batch_size: 48, created_at: 2025-08-21-135339	138737.32674433687
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers10, args.global_batch_size: 48, created_at: 2025-08-21-134749	154746.97733244646
    machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers8, args.global_batch_size: 48, created_at: 2025-08-21-135130	175552.08689111052
    ```

    - Corrections:

        ```bash
        175552 / 24 = 7315.5
        154746 / 24 = 6447.75
        138737 / 24 = 5,780.71
        125730 / 24 = 5,238.75
        113606 / 24 = 4733.58
        104714 / 24 = 4363.08
        ```

- SmolLM3 Model size:

    ```bash
    [smollm3-nLayers18][nlayers: 18] machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers18, args.global_batch_size: 48, created_at: 2025-08-21-131910	2.454792192
    [smollm3-nLayers16][nlayers: 16] machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers16, args.global_batch_size: 48, created_at: 2025-08-21-125937	2.298546176
    [smollm3-nLayers14][nlayers: 14] machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers14, args.global_batch_size: 48, created_at: 2025-08-21-140507	2.14230016
    [smollm3-nLayers12][nlayers: 12] machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers12, args.global_batch_size: 48, created_at: 2025-08-21-135339	1.986054144
    [smollm3-nLayers10][nlayers: 10] machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers10, args.global_batch_size: 48, created_at: 2025-08-21-134749	1.829808128
    [smollm3-nLayers8][nlayers: 8] machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: smollm3-nLayers8, args.global_batch_size: 48, created_at: 2025-08-21-135130 1.673562112
    ```

- Explicitly:

    ```bash
    [llama3-3B-nLayers14][nlayers: 14] machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers14, args.global_batch_size: 48, created_at: 2025-08-21-124405	2.982239232
    [llama3-3B-nLayers12][nlayers: 12] machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers12, args.global_batch_size: 48, created_at: 2025-08-21-133138	2.780900352
    [llama3-3B-nLayers12][nlayers: 12] machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers12, args.global_batch_size: 96, created_at: 2025-08-21-121138	2.780900352
    [llama3-3B-nLayers10][nlayers: 10] machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers10, args.global_batch_size: 48, created_at: 2025-08-21-134052	2.579561472
    [llama3-3B-nLayers10][nlayers: 10] machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers10, args.global_batch_size: 48, created_at: 2025-08-21-122810	2.579561472
    [llama3-3B-nLayers8][nlayers: 8] machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers8, args.global_batch_size: 48, created_at: 2025-08-21-134526	2.378222592
    [llama3-3B-nLayers6][nlayers: 6] machine: Aurora, env.NHOSTS: 2, env.MODEL_ARCH: llama3-3B-nLayers6, args.global_batch_size: 48, created_at: 2025-08-21-142250	2.176883712
    ```
