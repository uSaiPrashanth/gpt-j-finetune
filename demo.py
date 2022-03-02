import torch
import torch.nn as nn
import torch.distributed as dist
import os
from argparse import ArgumentParser

def demo(args):
    dist.init_process_group(
        backend='gloo',
        init_method='env://',
        world_size=args.nnodes,
        rank=args.rank,
    )
    print("Connected")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--rank",type=int)
    parser.add_argument("--nnodes",type=int)
    args = parser.parse_args()
    
    demo(args)