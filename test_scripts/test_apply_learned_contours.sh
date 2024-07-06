#!/bin/bash -e

# experiment on applying vocabulary from corpus A to corpus B
base_path="data/corpora/test_apply_learned_contour"
orig_path="data/corpora/test_midis_tpq12_r40_d48_v4_t8_240_8"
bpe_path="data/corpora/test_midis_tpq12_r40_d48_v4_t8_240_8_bpe4_ours_1.0"
test -d $base_path && rm -r $base_path
mkdir $base_path

# make sure bpe programs are new
make -C bpe all

./bpe/mnbpe --log \
    --apply "${bpe_path}/contour_vocab" \
    "$orig_path" "$base_path" \
    | tee "${base_path}/test_apply_learned_contour.log" -a

python3 plot_bpe_log.py \
    "${base_path}" \
    "${base_path}/test_apply_learned_contour.log"

cp "${bpe_path}/contour_vocab" "$base_path"
cp "${bpe_path}/paras" "$base_path"
cp "${bpe_path}/pathlist" "$base_path"
python3 make_arrays.py --bpe --debug \
    --log "${base_path}/test_apply_learned_contour.log" \
    --worker-number 32 -- "${base_path}"
