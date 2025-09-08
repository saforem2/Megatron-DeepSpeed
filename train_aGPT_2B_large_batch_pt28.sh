#!/bin/bash --login
#PBS -q prod
#PBS -A AuroraGPT
#PBS -j oe
#PBS -l walltime=06:00:00,filesystems=flare:home
#PBS -l select=256

# Get the directory of this script
whereami="$(dirname "$(realpath "$0")")"
cd "${whereami}" || {
	echo "Failed to change directory to ${whereami}"
	exit 1
}

setup_env() {
	PBS_O_WORKDIR="$(pwd)"
	export PBS_O_WORKDIR
	export http_proxy="http://proxy.alcf.anl.gov:3128"
	export https_proxy="http://proxy.alcf.anl.gov:3128"
	export ftp_proxy="http://proxy.alcf.anl.gov:3128"
	export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
	export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"

	# shellcheck disable=SC1090
	source /home/foremans/utils.sh

	conda_env=/flare/datascience/foremans/micromamba/envs/2025-07-pt28
	conda_name="$(basename "$conda_env")"
	dst_env="/tmp/${conda_name}"

	if [[ ! -d "${dst_env}" ]]; then
		echo "No cached conda environment found, creating a new one"
		ezpz_setup_env
		ezpz-yeet-env --src "${conda_env}.tar.gz"
		deactivate
		conda deactivate
	fi

	ezpz_load_new_pt_modules_aurora
	conda activate "${dst_env}"
	ezpz_setup_env
	log_message INFO "Using: $(which python3)"
}

train_model() {
	MODEL_ARCH=smollm3-3B \
		OPT=ipex.fusedlamb \
		NLAYERS=12 \
		GRAD_ACC_STEPS=2 \
		MICRO_BATCH=1 \
		USE_ACTIVATION_CHECKPOINTING=0 \
		ZERO_STAGE=0 \
		LR_DECAY_STYLE=constant \
		TOKENIZER_TYPE=HFTokenizer \
		TOKENIZER_MODEL=google/gemma-7b \
		DATA_FILE_LIST=ALCF/data-lists/aurora/olmo-mix-1124.txt \
		bash "${PBS_O_WORKDIR}/train_alcf.sh" \
		"$@"
}

main() {
	setup_env
	train_model "$@"
}

main "$@"
