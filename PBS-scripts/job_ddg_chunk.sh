#!/bin/bash
#PBS -o job_stdout.txt
#PBS -e job_errout.txt
#PBS -N ddg
#SBATCH -o job_stdout.txt
#SBATCH -e job_errout.txt
#SBATCH --job-name=ddg

# Arguments:
# infile
# outdir
# foldxversion

# Change directory for logging
# cd "${PBS_O_WORKDIR}"
echo "${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}"
echo $infile
echo $outdir
echo $foldxv

# Prepare environment
# Activate conda
. "${HOME}/miniconda3/etc/profile.d/conda.sh"
# Activate project environment
cd "${WORK}"
. activate.sh

# Activate model environment
# For GPU nodes:
# module load CUDA  # <- probably not necessary
# conda activate ppi_cuda
conda activate geoppi
cd "${ROOT_DIR}/models/GeoPPI"

# Prepare output directory and file
outpath="${outdir}/results.csv"
errdir="${outdir}/errors"
mkdir -p "${outdir}"
mkdir -p "${errdir}"
if [ ! -f "${outpath}" ]; then
    echo "Mutation;ddG;Python time;Real time;User time;Sys time" > "${outpath}"
fi

# Predict for each mutation in chunk
for mut in $(cat $infile)
do
  PBS-scripts/pred_ddg.sh $SAK_MPLASMIN_PDB $mut $outpath $errdir $foldxv &
done
wait

exit 0
