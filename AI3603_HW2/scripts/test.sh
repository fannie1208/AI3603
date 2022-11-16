#!/bin/bash
bsizes=(1000 2000 3000)
target_frequencies=(50 200 300 500)
start_e=(0.3 0.5 0.7 0.9 1)
end_e=(0 0.01 0.05)
exploration_fraction=(0.05 0.1 0.2 0.3)
for e2 in ${end_e[@]}
do
  for e1 in ${start_e[@]}
  do
    for f in ${exploration_fraction[@]}
    do
      CUDA_VISIBLE_DEVICES=3 python dqn.py --gamma 0.982 --buffer-size 2000 --target-network-frequency 50 --end-e ${e2} --start-e ${e1} --exploration-fraction ${f}
    done
  done
done

