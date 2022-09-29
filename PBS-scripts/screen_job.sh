#!/bin/bash
#PBS -o job_stdout.txt
#PBS -e job_errout.txt
#PBS -N ddg

# Parameters for manual run
# pdbfile="${1}"
# partners="${2}"
# mutfile="${3}"
# outdir="${4}"
# runname="${5}"
# wtkind="${6}"
# Example values
# "${SAK_MPLASMIN_PDB}"
# "${SAK_CHAINID}_${MPLASMIN_CHAINID}"
# "${MUTATIONS_DIR}/random-for-tests/1p/0"
# "${PREDICTIONS_DIR}/foldx-preprocessing-choice/test"
# "test-run"
# "foldx_byproduct"
# PBS Example
# qsub -A ${PROJECTID} -q qexp -v pdbfile="${SAK_MPLASMIN_PDB}",partners="${SAK_CHAINID}_${MPLASMIN_CHAINID}",mutfile="${MUTATIONS_DIR}/affilib-space/best_clustered_mutants-50/0",outdir="${PREDICTIONS_DIR}/GeoPPI/retrained/best_clustered_mutants-50",runname="prediction",wtkind="foldx_byproduct" screen_job.sh


# TODO set properly
stdoutfile="job_stdout.txt"
stderrfile="job_errout.txt"

# Change to working directory for logging
cd "${PBS_O_WORKDIR}" || exit 1

# Prepare project environment
. "${HOME}/miniconda3/etc/profile.d/conda.sh"
cd "${WORK}" || exit 2
. activate.sh

# Prepare model environment
conda activate geoppi
cd "${ROOT_DIR}/models/GeoPPI" || exit 3

# Construct run id
rundatetime=$(date +'%d-%m-%Y_%H-%M-%S')
runid="${runname}_${rundatetime}"

# Run screening
python screen.py "${pdbfile}" "${partners}" "${mutfile}" "${outdir}" "${runid}"\
                 --wt_kind "${wtkind}"

# Copy logging files
logsdir="${outdir}/logs"
if [ -s "${stdoutfile}" ]; then
  mkdir -p "${logsdir}"
  cp "${stdoutfile}" "${logsdir}"
fi
if [ -s "${stderrfile}" ]; then
  mkdir -p "${logsdir}"
  cp "${stderrfile}" "${logsdir}"
fi

