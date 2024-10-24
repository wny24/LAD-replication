#!/bin/bash

run_name="iconic-dust-2"
num_epochs=100

python tests/test_reconstruction.py +wandb.run_name=$run_name training.num_epochs=$num_epochs