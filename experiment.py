import argparse
from collections import defaultdict
import os
import pathlib
from time import perf_counter, sleep

import pandas as pd
from tqdm import tqdm

from BNReasoner import BNReasoner

"""
THIS IS A TESTING VERSION OF THE SCRIPT!

Lines to change after testing complete:
31 - remove [:3]
55 + 65 - replace with actual query
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to input folder", type=str, default='testing/test_set1/')
    parser.add_argument("--output", help="path to output folder", type=str, default='results.csv')
    # parser.add_argument("--iter", help="maximum amount of iterations", default=5, type=int)
    parser.add_argument("--heuristics", help="heuristic to use in ordering", default='min_deg', type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    stats = []
    for size_folder in tqdm(os.listdir(args.input)[:3]):
        if not size_folder.isdigit():
            continue
        
        for net_file in os.listdir(f'{args.input}{size_folder}'):
            net_path = f'{args.input}{size_folder}/{net_file}'
            print(f'Processing {net_path} file.')

            bayes_net = BNReasoner(net_path)

            # collect stats about pruned nodes and edges
            # motivation: we leave Q and e the same for pruned network
            num_nodes = bayes_net.bn.get_num_nodes()
            num_edges = bayes_net.bn.get_num_edges()
            
            # q_ratio = 30%, e_ratio = 20% [smaller precentage -> more prunning]
            Q, e = bayes_net.bn.rand_Qe(q_ratio=0.3, e_ratio=0.2)

            tic = perf_counter()
            sleep(0.2) # bayes_net.marginal_distribution(set(Q), e, args.heuristics)
            toc = perf_counter()
            time_before_prune = toc - tic

            bayes_net.prune(Q, e) 

            num_nodes_pruned = bayes_net.bn.get_num_nodes()
            num_edges_pruned= bayes_net.bn.get_num_edges()

            tic = perf_counter()
            sleep(0.1) # bayes_net.marginal_distribution(Q, e)
            toc = perf_counter()
            time_after_prune = toc - tic

            stats.append({
                'size': size_folder,
                'filename': net_file,
                'nodes_before_prune': num_nodes,
                'edges_before_prune': num_edges,
                'time_before_prune': time_before_prune,
                'nodes_after_prune': num_nodes_pruned,
                'edges_after_prune': num_edges_pruned,
                'time_after_prune': time_after_prune
            })

    print(f"Saving results to {args.output}")
    pd.DataFrame(stats).to_csv(args.output)
