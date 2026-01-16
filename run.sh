#!/usr/bin/env bash

# univaraite datasets
datasets="ECG EPG Gait NASA"
# multivariate datasets
#datasets="LTDB MITDB SVDB SMD"

tag="exp"
for data in ${datasets}; do
  echo ${data}
  nohup python3 main.py \
  --dataset ${data} \
  --tag ${tag} \
  --seed 42 123 2025 3407 7777 \
   > ${data}_${tag}_run.log 2>&1 &
done

tail -f ${data}_${tag}_run.log

