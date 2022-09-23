#!/bin/bash
#PBS -o train_job_stdout.txt
#PBS -e train_job_errout.txt
#PBS -N geoppi_train

stdoutfile="train_job_stdout.txt"
stderrfile="train_job_errout.txt"

# Change to working directory for logging
cd "${PBS_O_WORKDIR}" || exit 101

# Prepare project environment
. "${HOME}/miniconda3/etc/profile.d/conda.sh"
cd "${WORK}" || exit 102
. activate.sh

# Prepare model environment
conda activate geoppi
cd "${ROOT_DIR}/models/GeoPPI" || exit 103

python train.py

logsdir="${ROOT_DIR}/models/GeoPPI/trainedmodels/SKEMPI2-ML/rf"
if [ -s "${stdoutfile}" ]; then
  mkdir -p "${logsdir}"
  cp "${stdoutfile}" "${logsdir}"
fi
if [ -s "${stderrfile}" ]; then
  mkdir -p "${logsdir}"
  cp "${stderrfile}" "${logsdir}"
fi
