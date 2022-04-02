import os
import wandb
import sys
import signal
import subprocess

lr = 3e-3
for dropout in [0.0, 0.2, 0.5]:
    command = f"python run_lateral_model.py --dropout {dropout} --lr {lr}"
    os.system(command)

dropout=0.2
for lr in [3e-3, 3e-4, 3e-2, 1e-3, 1e-4]:
    command = f"python run_lateral_model.py --dropout {dropout} --lr {lr}"
    os.system(command)
