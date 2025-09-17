#!/bin/bash --login
# Example usage:
# WALLTIME="06:00:00" NJOBS=4 NHOSTS=64 bash train_aGPT_7B_chain.sh

NHOSTS="${NHOSTS:-24}"
#!/bin/bash --login
# Example usage:
# WALLTIME="06:00:00" NJOBS=4 NHOSTS=64 bash train_aGPT_7B_chain.sh

NHOSTS="${NHOSTS:-24}"
WALLTIME="${WALLTIME:-"02:00:00"}"
NJOBS="${NJOBS:-10}"

JOBIDS=()

printf "Caught NHOSTS: %s, WALLTIME: %s, NJOBS: %s\n" "${NHOSTS}" "${WALLTIME}" "${NJOBS}"
printf "Submitting job 0/%s\n" "${NJOBS}"
job_cmd="qsub -A datascience -q prod -l select=${NHOSTS} -l walltime=${WALLTIME},filesystems=flare:home ~/test.sh"
printf "JOB0: %s\n" "${job_cmd}"
# JOBIDS+=("$(bash -c "${job_cmd}")")
# JOBIDS+=("$(eval "${job_cmd}")")
# echo "${JOBIDS[@]}"
jobid=($(eval "${job_cmd}"))
echo "First jobid: ${jobid[*]}"

for (( idx=1; idx<="${NJOBS}"; idx+=1 )); do
    echo "Submitting job ${idx}/${NJOBS}"
    printf "JOBIDS[idx]: %s\n" "${JOBIDS[idx]}"
    echo "${JOBIDS[idx]}"
    job_cmd="qsub -W depend=afterany:${JOBIDS[idx]} -A datascience -q prod -l select=${NHOSTS} -l walltime=${WALLTIME},filesystems=flare:home ~/test.sh"
    echo "Job cmd for JOB ${idx}:"
    echo "${job_cmd}"
    JOBIDS+=("$(bash -c "${job_cmd}")")
    # printf "%s\n" "${job_cmd}"
    # eval "${job_cmd}"
    # JOBIDS+=("$(eval "${job_cmd}")")
    # printf "Submitted %s/%s: %s\n" "${idx}" "${NJOBS}" "${JOBIDS[((idx + 1))}"
done
