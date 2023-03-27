#!/bin/bash --login
#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃                GPT MODEL SETTINGS                   ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3 Small:  125M ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="125M"
# NLAYERS=12
# HIDDEN=768
# ATEN_HEADS=12
# GLOBAL_BATCH=512

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 1.5B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="1.5B"
# NLAYERS=48
# HIDDEN=1600
# ATEN_HEADS=25
# GLOBAL_BATCH=128

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 1.5B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="1.5B"
# NLAYERS=48
# HIDDEN=1600
# ATEN_HEADS=25
# GLOBAL_BATCH=128

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 2.7B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
MODEL_SIZE="2.7B"
NLAYERS=32
HIDDEN=2560
ATEN_HEADS=32
GLOBAL_BATCH=512

# ┏━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ ✓ GPT-3: 6.7B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="6.7B"
# NLAYERS=32
# HIDDEN=4096
# ATEN_HEADS=32
# GLOBAL_BATCH=1024

# ┏━━━━━━━━━━━━━━━━━━━━━┓
# ┃ ✓ GPT-3: 13B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="13B"
# NLAYERS=40
# HIDDEN=5120
# ATEN_HEADS=40
# GLOBAL_BATCH=1024

# ┏━━━━━━━━━━━━━━━━━━━━━┓
# ┃ ✓ GPT-3: 20B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="20B"
# NLAYERS=44
# HIDDEN=6144
# ATEN_HEADS=64
# GLOBAL_BATCH=1024

# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 25B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="25B"
# NLAYERS=64
# HIDDEN=5760
# ATEN_HEADS=64
# GLOBAL_BATCH=1024

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 175B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="175B"
# NLAYERS=96
# HIDDEN=12288
# ATEN_HEADS=96
# GLOBAL_BATCH=1536


export MODEL_SIZE="${MODEL_SIZE}"
export NLAYERS="${NLAYERS}"
export HIDDEN="${HIDDEN}"
export ATEN_HEADS="${ATEN_HEADS}"
export GLOBAL_BATCH="${GLOBAL_BATCH}"