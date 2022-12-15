import argparse
from collections import defaultdict
import os
from time import perf_counter
from datetime import timedelta

import pandas as pd
from tqdm import tqdm

from BNReasoner import BNReasoner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to input folder", type=str, default='testing/test_set1/')
    parser.add_argument("--output", help="path to output folder", type=str, default='results.csv')
    parser.add_argument("--q_ratio", help="the fraction of variables in each query", type=float, default=0.3)
    parser.add_argument("--e_ratio", help="the fraction of variables in the evidence", type=float, default=0.2)
    # parser.add_argument("--iter", help="maximum amount of iterations", default=5, type=int)
    parser.add_argument("--heuristics", help="heuristic to use in ordering", default='min_deg', type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    stats = []
    start_time = perf_counter()
    for size_folder in tqdm(os.listdir(args.input)):
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
            Q, e = bayes_net.bn.rand_Qe(q_ratio=args.q_ratio, e_ratio=args.e_ratio)

            tic = perf_counter()
            bayes_net.marginal_distribution(set(Q), e)
            toc = perf_counter()
            time_before_prune = toc - tic

            bayes_net.prune(Q, e) 

            num_nodes_pruned = bayes_net.bn.get_num_nodes()
            num_edges_pruned= bayes_net.bn.get_num_edges()

            tic = perf_counter()
            bayes_net.marginal_distribution(set(Q), e)
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

    end_time = perf_counter()
    seconds_elapsed = end_time - start_time

    print("Done!")
    print(f"Elapsed Time = {timedelta(seconds=int(seconds_elapsed))}")
    print(f"Saving results to {args.output}")
    pd.DataFrame(stats).to_csv(args.output)
