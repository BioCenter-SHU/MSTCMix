#!/bin/bash    
seeds=(" " ) 

for seed in "${seeds[@]}"; do
    python Train.py \
        --seed "$seed" \
        --batch_size  32\
        --num_epochs 50 \
	--num_warmup 5\
	--r 0.6\
        --gpu 0,1,2,3,4,5,6,7\
        2>> stderr.log
done