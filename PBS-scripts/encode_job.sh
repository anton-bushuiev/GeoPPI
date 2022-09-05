#!/bin/bash
#PBS -o job_stdout.txt
#PBS -e job_errout.txt
#PBS -N geoppi_encode

# Arguments
# infile
# outdir
# Run example
# qsub -A ${PROJECTID} -q qexp -v infile="${SKEMPI2_DIR}/misc/GeoPPI/inputs/0",outdir="${SKEMPI2_DIR}/misc/GeoPPI/features" encode_job.sh

# Change to working directory for logging
cd "${PBS_O_WORKDIR}" || exit 1

# Prepare project environment
. "${HOME}/miniconda3/etc/profile.d/conda.sh"
cd "${WORK}" || exit 2
. activate.sh

# Prepare model environment
conda activate geoppi
cd "${ROOT_DIR}/models/GeoPPI" || exit 3

while read inparams
do
  # Run encoding
  (python encode.py ${inparams} --outdir="${outdir}") &
done < "${infile}"
wait
