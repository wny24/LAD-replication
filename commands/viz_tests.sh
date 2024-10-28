#!/bin/bash

run_name="spring-night-11"
num_epochs=200
test_file="/mnt/data/erbauer/retargeting/retargeted_data_v4_faiveonly_plush_pick_test.npy"

python tests/test_reconstruction.py +wandb.run_name=$run_name +data.test_file=$test_file data.test_episode=true 
python tests/test_tsne_embeddings.py +wandb.run_name=$run_name +data.test_file=$test_file data.test_episode=true 
python tests/test_temporal_tsne.py +wandb.run_name=$run_name +data.test_file=$test_file data.test_episode=true
python tests/test_cross_reconstruction.py +wandb.run_name=$run_name +data.test_file=$test_file data.test_episode=true
