#!/bin/bash --login

#####################################
# AuroraGPT-7B
#
# Main production script for training
# AuroraGPT-7B @ ALCF
#####################################

# 1. Navigate into `$PBS_O_WORKDIR`
cd "${PBS_O_WORKDIR}" || exit
HERE=$(python3 -c 'import os; print(os.getcwd())') && export HERE
# 2. source `ALCF/helpers.sh`
source "${HERE}/ALCF/helpers.sh" || exit
# 3. call `setup` from `./ALCF/helpers.sh`
setup "$@" || exit
# # 4. Take custom args
# export custom_args=" $@"
# 5. Update ${run_cmd} (from setup ALCF/helpers.sh) with ${custom_args}
export run_cmd="${run_cmd}"
# 6. Add "${run_cmd}" to output log
echo "${run_cmd}" | tee -a "${OUTPUT_LOG}"
# 7. Tell user where to find output
printf "[!! %s] View output at:\n %s\n" "$(printBlue "NOTE")" "$(printYellow "${OUTPUT_LOG}")" | tee -a "${OUTPUT_LOG}"
# 8. Evaluate ${run_cmd} and append outputs to ${OUTPUT_LOG}
eval "${run_cmd}" |& tee -a "${OUTPUT_LOG}"
set +x
