#!/bin/bash --login

# 1. Source `ezpz/bin/uitils.sh` and setup {job, python} environment:
# NO_COLOR=1 source <(curl -sL https://bit.ly/ezpz-utils) && ezpz_setup_env
script -efq /dev/null -c "source <(curl -sL https://bit.ly/ezpz-utils) && ezpz_setup_env"
# script  -q /dev/null source <(curl -sL https://bit.ly/ezpz-utils) && ezpz_setup_env

# 2. Source `ALCF/helpers.sh` for Megatron-DeepSpeed setup
source "ALCF/helpers.sh" || exit

# 3. Call `setup` from `./ALCF/helpers.sh`
setup "$@" || exit

# 4. Run:
echo "${run_cmd[@]}" | tee -a "${OUTPUT_LOG}"
eval "${run_cmd[*]}" 2>&1 | tee -a "${OUTPUT_LOG}"
