import argparse
from collections import defaultdict
import os
import pathlib
from time import time

import pandas as pd
from tqdm import tqdm

from BNReasoner import BNReasoner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to input folder", type=str, default='testing/test_set1/')
    parser.add_argument("--output", help="path to output folder", type=str, default='results.xlsx')
    # parser.add_argument("--iter", help="maximum amount of iterations", default=5, type=int)
    parser.add_argument("--heuristics", help="heuristic to use in ordering", default='min_deg', type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    stats = []
    # iterate over files, not sizes
    for size_folder in tqdm(os.listdir(args.input)):
        for net_file in os.listdir(f'{args.input}{size_folder}'):
            net_path = f'{args.input}{size_folder}/{net_file}'
            print(f'Processing {net_path} file.')
            bayes_net = BNReasoner(net_path)


            # collect stats about prunned nodes and edges
            # motivation: we leave Q and e the same for prunned network
            node_amt = bayes_net.bn.get_num_nodes()
            edges_amt = bayes_net.bn.get_num_edges()
            
            Q, e = bayes_net.bn.rand_Qe(q_ratio=0.3, e_ratio=0.2) # q_ratio = 30%, e_ratio = 20% [smaller precentage -> more prunning]

            start_time = time()
            bayes_net.marginal_distribution(set(Q), e, args.heuristics)
            finish_time = time()

            bayes_net.prune(Q, e) 

            node_amt_prunned = bayes_net.bn.get_num_nodes()
            edges_amt_prunned= bayes_net.bn.get_num_edges()

            start_time_prunned = time()
            bayes_net.marginal_distribution(Q, e)
            finish_time_prunned = time()

            stats.append({
                'size': size_folder,
                'filename': net_file,
                'node_amt': node_amt,
                'edges_amt': edges_amt,
                'start_time': start_time,
                'finish_time': finish_time,
                'node_amt_prunned': node_amt_prunned,
                'edges_amt_prunned': edges_amt_prunned,
                'start_time_prunned': start_time_prunned,
                'finish_time_prunned': finish_time_prunned,
            })

    print('Saving results.')
    pd.DataFrame(stats).to_excel(args.output)
