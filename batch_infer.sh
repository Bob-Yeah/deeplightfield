#/usr/bin/bash

testcase=$1
dataset='data/sp_view_syn_2020.12.31_fovea'
epochs=100

n_layers_arr=(4 8 4 8)
n_samples_arr=(16 16 32 32)
for nf in 64 128 256; do
    n_layers=${n_layers_arr[$testcase]}
    n_samples=${n_samples_arr[$testcase]}
    configid="infer_test@msl-rgb_e10_fc${nf}x${n_layers}_d1-50_s${n_samples}"
    python run_spherical_view_syn.py --dataset $dataset/train.json --config-id $configid --device $testcase --epochs $epochs
    python run_spherical_view_syn.py --dataset $dataset/train.json --test $dataset/$configid/model-epoch_$epochs.pth --perf --device $testcase
    python run_spherical_view_syn.py --dataset $dataset/test.json --test $dataset/$configid/model-epoch_$epochs.pth --perf --device $testcase
done
