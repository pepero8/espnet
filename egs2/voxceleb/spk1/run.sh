#!/usr/bin/env bash
set -e
set -u
set -o pipefail


# spk_config=conf/train_RawNet3.yaml
spk_config=conf/tuning/train_ECAPA_mel_jhwan.yaml

train_set="voxceleb12_devs"
valid_set="voxceleb1_test"
cohort_set="voxceleb2_test"
test_sets="voxceleb1_test"
feats_type="raw"

./spk.sh \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --cohort_set ${cohort_set} \
    --test_sets ${test_sets} \
    "$@"
