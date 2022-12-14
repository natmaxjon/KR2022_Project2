import argparse
from collections import defaultdict
from time import time

from BNReasoner import BNReasoner
from gen_test_set import gen_bns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to input folder", type=str)
    parser.add_argument("--output", help="path to output folder", type=str)
    parser.add_argument("--max_size", help="maximum size for the BN", default=10, type=int)
    parser.add_argument("--iter", help="maximum amount of iterations", default=5, type=int)
    parser.add_argument("--heuristics", help="heuristic to use in ordering", default='min_deg', type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    stats = {}

    # iterate over files, not sizes
    for size in range(args.max_size):
        stats['bare_net'] = defaultdict(list)
        stats['prunned_net'] = defaultdict(list)

        gen_bns(size)
        bayes_net = BNReasoner(args.input)
        # query = generate_query()
        query = None
        for iter in range(args.iter):
            start_time = time()
            # run query on bn
            finish_time = time()
            stats['bare_net'][size].append(finish_time - start_time)

        # consider iteration over each baesian network for new query

        bayes_net.prune(**query) # q_ratio = 30%, e_ratio = 20% [smaller precentage -> more prunning]
        # collect stats about prunned nodes and edges
        # motivation: we leave Q and e the same for prunned network

        for iter in range(args.iter):
            start_time = time()
            # run query on bn
            finish_time = time()
            stats['prunned_net'][size].append(finish_time - start_time)

    # save stats as pd DataFrame to output folder

