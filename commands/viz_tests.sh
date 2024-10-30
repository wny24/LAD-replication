#!/bin/bash

run_name="tricky-twilight-40"
num_epochs=310
test_file="/mnt/data/erbauer/retargeting/retargeted_data_v4_faiveonly_plush_pick_test.npy"
test_file_h5="/mnt/data1/erbauer/plush_pick_test/episode_0_success.h5"

python tests/test_reconstruction.py +wandb.run_name=$run_name +data.test_file=$test_file data.test_episode=true +training.test_epoch=$num_epochs
python tests/test_reconstruction_h5.py +wandb.run_name=$run_name +data.test_file=$test_file_h5 data.test_episode=true +training.test_epoch=$num_epochs
python tests/test_tsne_embeddings.py +wandb.run_name=$run_name +data.test_file=$test_file data.test_episode=true +training.test_epoch=$num_epochs
python tests/test_temporal_tsne.py +wandb.run_name=$run_name +data.test_file=$test_file data.test_episode=true +training.test_epoch=$num_epochs
python tests/test_cross_reconstruction.py +wandb.run_name=$run_name +data.test_file=$test_file data.test_episode=true +training.test_epoch=$num_epochs
