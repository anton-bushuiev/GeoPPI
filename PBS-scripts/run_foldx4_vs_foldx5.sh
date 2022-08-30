#!/bin/bash
for foldxv in "4" "5"
do
  for indir in "1p" "2p" "3p" "7p" "10p"
  do
    qsub -v infile="${MUTATIONS_DIR}/random-for-tests/${indir}/0",outdir="${DATA_DIR}/predictions/foldx4-vs-foldx5/foldx${foldxv}",foldxv="${foldxv}" -A OPEN-26-23 -q qexp job_ddg_chunk.sh
  done
done
