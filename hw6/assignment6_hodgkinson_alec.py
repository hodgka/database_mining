import numpy as np
import argparse
import re
import networkx

np.set_printoptions(threshold=np.inf, precision=4, suppress=True)


def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('--filename')
    args.add_argument('--minsup', nargs='?')
    return args.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
