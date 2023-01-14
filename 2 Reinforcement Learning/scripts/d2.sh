#!/bin/bash
lrs=(1e-3 1e-2)
bsizes=(1000 5000 10000 20000 50000)
gammas=(0.7 0.8 0.9 0.98 0.99 1)
target_frequencies=(10 100 500 750 1000)
for lr in ${lrs[@]}
do
  for bsize in ${bsizes[@]}
  do
    for gamma in ${gammas[@]}
    do
      for f in ${target_frequencies[@]}
      do 
      CUDA_VISIBLE_DEVICES=0 python dqn.py --learning-rate ${lr} --buffer-size ${bsize} --gamma ${gamma} --target-network-frequency ${f}
      done
    done
  done
done