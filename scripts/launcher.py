#!/usr/bin/env python3

import argparse
import logging
import os
import crypten
import multiprocessing
import uuid
import cProfile
import crypten.communicator as comm
import benchmark
from benchmark import *

parser = argparse.ArgumentParser()
experiments = ['train_all', 'inference_all', 'train_all_plaintext', 'inference_all_plaintext', 'batch_inference']


use_csprng = True
sync_key = False

def run_experiment(args):
    level = logging.INFO
    logging.getLogger().setLevel(level)

    crypten.init(use_csprng=use_csprng, sync_key=sync_key)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    )

    # Select the benchmark experiment to run. 
    train_all()
    # inference_all()
    # train_all_plaintext()
    # inference_all_plaintext()
    # batch_inference()


if __name__ == "__main__":
    args = parser.parse_args()
    run_experiment(args)


    

