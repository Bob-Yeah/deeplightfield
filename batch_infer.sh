#/usr/bin/bash

testcase=$1
dataset='data/gas_fovea_r90x30_t0.3_2021.01.11'
epochs=50

# layers: 4, 8
# samples: 4, 16, 64
# channels: 64 128 256
x=0
nf_arr=($x 128 256 256)
n_layers_arr=($x 8 4 8)
n_samples=32
nf=${nf_arr[$testcase]}
n_layers=${n_layers_arr[$testcase]}
#for n_layers in 4 8; do
#    for nf in 64 128 256; do
#        for n_samples in 4 16 64; do
            configid="infer_test@msl-rgb_e10_fc${nf}x${n_layers}_d1.00-50.00_s${n_samples}"
            python run_spherical_view_syn.py --dataset $dataset/train.json --config-id $configid --device $testcase --epochs $epochs
            python run_spherical_view_syn.py --dataset $dataset/train.json --test $dataset/$configid/model-epoch_$epochs.pth --perf --device $testcase
            python run_spherical_view_syn.py --dataset $dataset/test.json --test $dataset/$configid/model-epoch_$epochs.pth --perf --device $testcase
#        done
#    done
#done
