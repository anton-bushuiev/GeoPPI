#!/bin/bash
#PBS -q qlong
#PBS -l select=1,walltime=00:10:00
#PBS -o ddg_job_stdout.txt
#PBS -e ddg_job_errout.txt

# For manual run (not via qsub):
# mutation=$1
# pdbfile=$2
# outdir=$3

# Change directory for logging
cd $PBS_O_WORKDIR

# Prepare environment
module load CUDA
. "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate ppi # For GPU nodes: conda activate ppi_cuda
cd "${HOME_SAK}/models/GeoPPI"

# Prepare output directory and file
outpath="${outdir}/results.csv"
errdir="${outdir}/errors"
mkdir -p "$outdir"
mkdir -p "$errdir"
if [ ! -f "$outpath" ]; then
    echo "Mutation;ddG;Python time;Real time;User time;Sys time" > "$outpath"
fi

# Run and store prediction with elapsed time into variable
out=$((export TIMEFORMAT="%R,%U,%S" ; time python run.py \
    "$pdbfile" \
    "$mutation" \
    "A_C" \
) 2>&1)

# Check output and write save results in .csv format
fl=[-+]?[0-9]+\.?[0-9]*
if [[ $out  =~ ^${fl},${fl},${fl},${fl},${fl}$ ]]
then
    echo "${mutation},${out}" >> "$outpath"
else
    echo "${mutation},,,," >> "$outpath"
    echo "$out" >> "${errdir}/${mutation}.err"
fi
