import argparse
import random

def build_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=str,help="model name")
    parser.add_argument("--dataset",type=str,help="training dataset name")
    ...
    return parser.parse_args()