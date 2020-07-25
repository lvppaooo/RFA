#!/usr/bin/env bash

#################  FedAvg ######################
## corrupt: todo

# outf="outputs_femnist_cnn/outputs_femnist_cnn_avg"
# logf="outputs_femnist_cnn/log_femnist_cnn_avg"
# num_rounds=500
# batch_size=64
# num_epochs=5
# clients_per_round=20
# lr=5e-2
# lrd=2

# common=" --num-rounds ${num_rounds} --eval-every 10 --clients-per-round ${clients_per_round} --batch_size ${batch_size} --num_epochs ${num_epochs} -lr $lr --lr-decay $lrd  "


# for seed in 3 4 5
# do
#     time python models/main.py -dataset femnist -model cnn $common --seed $seed  --output_summary_file ${outf}_clean_${seed}  > ${logf}_clean_${seed} 2>&1
#     for frac in 0.1
#     do
#         for corruption in p_x omniscient
#         do
#                 time python models/main.py -dataset femnist -model cnn $common --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${corruption}_${frac}_${seed}  > ${logf}_${corruption}_${frac}_${seed} 2>&1
#         done
#     done
# done


# ####################  RobustFedAvg  ######################
# outf="outputs_femnist_cnn/outputs_femnist_cnn_rfa"
# logf="outputs_femnist_cnn/log_femnist_cnn_rfa"
# num_rounds=500
# batch_size=64
# num_epochs=5
# clients_per_round=20
# lr=5e-2
# lrd=2
# niter=2

# common=" --num-rounds ${num_rounds} --eval-every 10 --clients-per-round ${clients_per_round} --batch_size ${batch_size} --num_epochs ${num_epochs} -lr $lr --lr-decay $lrd  --aggregation geom_median --weiszfeld-maxiter ${niter}"


# for seed in 1 2 3
# do
#     time CUDA_VISIBLE_DEVICES=0 python models/main.py -dataset femnist -model cnn $common --seed $seed  --output_summary_file ${outf}_clean_${seed}  > ${logf}_clean_${seed} 2>&1
#     for frac in 0.1 0.25
#     do
#         for corruption in p_x omniscient
#         do
#             time python models/main.py -dataset femnist -model cnn $common --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${corruption}_${frac}_${niter}_${seed}  > ${logf}_${corruption}_${frac}_${niter}_${seed} 2>&1
#         done
#     done
# done


# #################### Minibatch SGD ######################
# outf="outputs/outputs_femnist_cnn_mb"
# logf="outputs/log_femnist_cnn_mb"
# num_rounds=2000
# minibatch=0.1
# clients_per_round=20
# eval_every=80
# lr=0.2
# lrd=1

# main_args="-dataset femnist -model cnn"
# common=" --num-rounds ${num_rounds} --eval-every ${eval_every} --clients-per-round ${clients_per_round} --minibatch $minibatch -lr $lr --lr-decay $lrd  "


# for seed in 1 2 3 4 5
# do

# 	corruption=clean
# 	time python models/main.py ${main_args}  $common --seed $seed --output_summary_file ${outf}_${corruption}_${seed} > ${logf}_${corruption}_${seed} 2>&1
# 	for corruption in p_x omniscient
# 	do
#         for frac in 0.01 0.25
#         do
#             time python models/main.py ${main_args}  $common --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${corruption}_${frac}_${seed}  > ${logf}_${corruption}_${frac}_${seed} 2>&1
#         done
# 	done


# done

############### Gaussian Federated #########################################################

outf="outputs_femnist_cnn/outputs_femnist_cnn_gs"
logf="outputs_femnist_cnn/log_femnist_cnn_gs"
num_rounds=200
batch_size=64
num_epochs=20
clients_per_round=20
lr=5e-2
lrd=2

common=" --num-rounds ${num_rounds} --eval-every 10 --clients-per-round ${clients_per_round} --batch_size ${batch_size} --num_epochs ${num_epochs} -lr $lr --lr-decay $lrd  --aggregation gaussian"


for seed in 1
do
    # time python models/main.py -dataset femnist -model cnn $common --seed $seed  --output_summary_file ${outf}_clean_${seed}  > ${logf}_clean_${seed} 2>&1
    # time python models/main.py -dataset femnist -model cnn $common --seed $seed  --output_summary_file ${outf}_clean_${seed} 
    #  > ${logf}_clean_${seed}
    for frac in 0.25 0.01
    do
        for corruption in p_x
        do
            time python models/main.py -dataset femnist -model cnn $common --seed $seed --corruption $corruption --fraction-corrupt $frac --output_summary_file ${outf}_${corruption}_${frac}_${niter}_${seed}  > ${logf}_${corruption}_${frac}_${niter}_${seed} 2>&1
        done
    done
done