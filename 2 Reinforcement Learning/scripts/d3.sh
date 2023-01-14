#!/bin/bash
bs=(64 128 256 300)
gammas=(0.98 0.99)
train_frequency=(10)

for b in ${bs[@]}
do
  for gamma in ${gammas[@]}
  do
    for f in ${train_frequency[@]}
    do
    CUDA_VISIBLE_DEVICES=3 python dqn.py --batch-size ${b} --gamma ${gamma} --train-frequency ${f}
    done
  done
done
