#!/usr/bin/env bash

# This script is used to associate indices
# to training to testing datasets
ORIGINAL_FOLDER="original"
SEED=42 # used for repeatability of the shuffling
N=3000 # number of pairs of files

# Size of each set
N_TRAIN=2700
N_TEST=300

# Outputs file containing indices for each set
FILE_TRAIN_INDICES="train_indices"
FILE_TEST_INDICES="test_indices"


# To have a repeatable shuffling
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

# We get the indices, with make then unique, then we shuffle them using a seed
ls $ORIGINAL_FOLDER | \
cut -d '_' -f1 |\
uniq |\
shuf --random-source=<(get_seeded_random $SEED)|\
head -$N_TRAIN > $FILE_TRAIN_INDICES

ls $ORIGINAL_FOLDER | \
cut -d '_' -f1 |\
uniq |\
shuf --random-source=<(get_seeded_random $SEED)|\
tail -$N_TEST > $FILE_TEST_INDICES
