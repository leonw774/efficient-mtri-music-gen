#!/bin/bash

# experiment on applying vocabulary from corpus A to corpus B
base_path="data/corpora/experiment_apply_learned_bpe_on_other"
lmd_orig_path="data/corpora/lmd_full_tpq12_r40_d48_v4_t8_240_8"
lmd_bpe_path="data/corpora/lmd_full_tpq12_r40_d48_v4_t8_240_8_bpe128_ours_1.0"
snd_orig_path="data/corpora/snd_tpq12_r40_d48_v4_t8_240_8"
snd_bpe_path="data/corpora/snd_tpq12_r40_d48_v4_t8_240_8_bpe128_ours_1.0"
rm -r $base_path
mkdir $base_path

worker_number=32

# make sure bpe programs are new
make -C ./bpe

## vocab source -- applied corpus

mkdir "${base_path}/lmd--snd_ours_1.0"
./bpe/mnbpe --log \
    --apply "${lmd_bpe_path}/contour_vocab" --worker-number $worker_number \
    "$snd_orig_path" "${base_path}/lmd--snd_ours_1.0" \
    | tee "${base_path}/lmd--snd_ours_1.0.log" -a

python3 plot_bpe_log.py \
    "${base_path}/lmd--snd_ours_1.0" \
    "${base_path}/lmd--snd_ours_1.0.log"

cp "${lmd_bpe_path}/contour_vocab" "${base_path}/lmd--snd_ours_1.0"
cp "${snd_orig_path}/paras"      "${base_path}/lmd--snd_ours_1.0"
cp "${snd_orig_path}/pathlist"   "${base_path}/lmd--snd_ours_1.0"
python3 make_arrays.py --bpe --debug \
    --log "${base_path}/lmd--snd_ours_1.0.log" \
    --worker-number $worker_number -- "${base_path}/lmd--snd_ours_1.0"


mkdir "${base_path}/snd--lmd_ours_1.0"
./bpe/mnbpe --log \
    --apply "${snd_bpe_path}/contour_vocab" --worker-number $worker_number \
    $lmd_orig_path "${base_path}/snd--lmd_ours_1.0" \
    | tee "${base_path}/snd--lmd_ours_1.0.log" -a

python3 plot_bpe_log.py \
    "${base_path}/snd--lmd_ours_1.0" \
    "${base_path}/snd--lmd_ours_1.0.log"

cp "${snd_bpe_path}/contour_vocab" "${base_path}/snd--lmd_ours_1.0"
cp "${lmd_orig_path}/paras"      "${base_path}/snd--lmd_ours_1.0"
cp "${lmd_orig_path}/pathlist"   "${base_path}/snd--lmd_ours_1.0"
python3 make_arrays.py --bpe --debug \
    --log "${base_path}/snd--lmd_ours_1.0.log" \
    --worker-number $worker_number -- "${base_path}/snd--lmd_ours_1.0"
