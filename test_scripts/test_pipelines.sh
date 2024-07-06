#!/bin/bash -e

# data preprocess

# make original corpus first
test -d "data/corpora/test_midis_tpq12_r40_d48_v4_t8_240_8" \
    && rm -r "data/corpora/test_midis_tpq12_r40_d48_v4_t8_240_8"
bash -e ./pipeline.sh test_midis no_bpe no_train --use-existed

# make multi-note bpe compressed corpus
test -d "data/corpora/test_midis_tpq12_r40_d48_v4_t8_240_8_bpe4_ours_1.0" \
    && rm -r "data/corpora/test_midis_tpq12_r40_d48_v4_t8_240_8_bpe4_ours_1.0"
bash -e ./pipeline.sh test_midis test_bpe no_train --use-existed

# experiment on sample rate
test -d "data/corpora/test_midis_tpq12_r40_d48_v4_t8_240_8_bpe4_ours_0.01" \
    && rm -r "data/corpora/test_midis_tpq12_r40_d48_v4_t8_240_8_bpe4_ours_0.01"
bash -e ./pipeline.sh test_midis test_bpe_sample0.01 no_train --use-existed

# train and eval

# full model
bash -e ./pipeline.sh test_midis test_bpe test_cpu --use-existed
