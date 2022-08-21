#!/bin/bash

# Example:
# PBS-scripts/pred_ddg.sh $SAK_MPLASMIN_PDB YC17T ./tmp-res/res.csv ./tmp-res 4

# Parse arguments
pdbfile=$1
mutation=$2
outpath=$3
errdir=$4
foldxversion=$5

# Run and store prediction with elapsed time into variable
out=$((export TIMEFORMAT="%R;%U;%S" ; time python run.py \
    "$pdbfile" \
    "$mutation" \
    "A_C" \
    "0" \
    "$foldxversion"
) 2>&1)

# Check output and write results in .csv format
fl=[-+]?[0-9]+\.?[0-9]*
if [[ $out  =~ ^${fl}\;${fl}\;${fl}\;${fl}\;${fl}$ ]]
then
    echo "${mutation};${out}" >> "$outpath"
else
    echo "${mutation};;;;" >> "$outpath"
    echo "$out" >> "${errdir}/${mutation}.txt"
fi
