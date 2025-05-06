#!/bin/bash --login
#PBS -q lustre_scaling
#PBS -A Aurora_Deployment
#PBS -j oe

#####################################
# AuroraGPT-7B
#
# Main production script for training
# AuroraGPT-7B @ ALCF
#####################################
train_aGPT() {

    # 1. Navigate into `$PBS_O_WORKDIR`
    # cd "${PBS_O_WORKDIR}" || exit
    HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
    GIT_BRANCH=$(git branch --show-current) && export GIT_BRANCH

    # 3. source `ezpz/bin/uitils.sh` and setup {job, python} environment:
    source <(curl 'https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh') 

    if [[ "${HERE}" != "${PBS_O_WORKDIR:-}" ]]; then
        export PBS_O_WORKDIR="${HERE}"
        printf "[!! %s] WARNING: Current working directory (%s) does not match PBS_O_WORKDIR (%s)\n" "$(printRed "WARNING")" "${HERE}" "${PBS_O_WORKDIR}"
        printf "[!! %s] This may cause issues with the job submission.\n" "$(printRed "WARNING")"
        printf "Setting PBS_O_WORKDIR to %s and continuing...\n" "${HERE}"
    fi

    ezpz_setup_env || exit

    # if  command -v "ezpz-test"; then
    #     printf "[!! %s] ezpz is already installed.\n" "$(printGreen "INFO")"
    # else
    #     printf "[!! %s] ezpz is not installed. Installing...\n" "$(printRed "WARNING")"
    #     python3 -m pip install "git+https://github.com/saforem2/ezpz"
    # fi

    # 2. source `ALCF/helpers.sh` for Megatron-DeepSpeed setup
    source "${HERE}/ALCF/helpers.sh" || exit

    # 3. call `setup` from `./ALCF/helpers.sh`
    setup "$@" || exit
    # export run_cmd="${run_cmd}"
    echo "${run_cmd[@]}" | tee -a "${OUTPUT_LOG}"

    # 4. Tell user where to find output
    printf "[!! %s] View output at:\n %s\n" "$(printBlue "NOTE")" "$(printYellow "${OUTPUT_LOG}")" | tee -a "${OUTPUT_LOG}"

    # 6. Evaluate ${run_cmd} and append outputs to ${OUTPUT_LOG}
    # eval "${run_cmd[*]}" |& tee -a "${OUTPUT_LOG}"
    if [[ "${DEBUG:-}" ]]; then
        set -x
        bash -c "${run_cmd[*]}" |& tee -a "${OUTPUT_LOG}"
        set +x
    else
        bash -c "${run_cmd[*]}" |& tee -a "${OUTPUT_LOG}"
    fi
}

train_aGPT "$@"
