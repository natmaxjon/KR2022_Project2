import argparse
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------- Settings --------------------------------- #

before_colour = '#d62728'
after_colour = '#1f77b4'
node_perc_colour = '#ff7f0e'
edge_perc_colour = '#17becf'
alpha = 0.15
linewidth = 2
legend_size = 15
axis_font_size = 15
tick_label_size = 13

# ---------------------------- Helper Functions ----------------------------- #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to results CSV file", type=str, default='results.csv')
    args = parser.parse_args()
    return args

def perc_change(before, after):
    if before == 0:
        return 0
        
    return 100 * (before - after) / before

# ---------------------------------- Main ----------------------------------- #

if __name__ == '__main__':
    args = parse_args()

    # Preprocessing
    stats = pd.read_csv(args.input)
    stats = stats.sort_values(by=['size'])#.head(140) # Uncomment to restrict plot to sizes in range [3, 16]
    stats['perc_nodes_pruned'] = stats.apply(lambda x: perc_change(x['nodes_before_prune'], x['nodes_after_prune']), axis=1)
    stats['perc_edges_pruned'] = stats.apply(lambda x: perc_change(x['edges_before_prune'], x['edges_after_prune']), axis=1)

    # Runtime before pruning
    avg_time_before = stats.groupby(['size'])['time_before_prune'].mean().to_numpy()
    std_time_before = stats.groupby(['size'])['time_before_prune'].std().to_numpy()
    time_before_plus_std = [a + b for a, b in zip(avg_time_before, std_time_before)]
    time_before_minus_std = [a - b for a, b in zip(avg_time_before, std_time_before)]

    # Runtime after pruning
    avg_time_after = stats.groupby(['size'])['time_after_prune'].mean().to_numpy()
    std_time_after = stats.groupby(['size'])['time_after_prune'].std().to_numpy()
    time_after_plus_std = [a + b for a, b in zip(avg_time_after, std_time_after)]
    time_after_minus_std = [a - b for a, b in zip(avg_time_after, std_time_after)]

    # Percentage node pruning
    avg_node_perc = stats.groupby(['size'])['perc_nodes_pruned'].mean().to_numpy()
    std_node_perc = stats.groupby(['size'])['perc_nodes_pruned'].std().to_numpy()
    node_perc_plus_std = [a + b for a, b in zip(avg_node_perc, std_node_perc)]
    node_perc_minus_std = [a - b for a, b in zip(avg_node_perc, std_node_perc)]

    # Percentage edge pruning
    avg_edge_perc = stats.groupby(['size'])['perc_edges_pruned'].mean().to_numpy()
    std_edge_perc = stats.groupby(['size'])['perc_edges_pruned'].std().to_numpy()
    edge_perc_plus_std = [a + b for a, b in zip(avg_edge_perc, std_edge_perc)]
    edge_perc_minus_std = [a - b for a, b in zip(avg_edge_perc, std_edge_perc)]

    # Extract sizes for x-axis
    sizes = stats['size'].unique()

    # --------- Plot Data --------- #
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # --- Runtime --- #
    # Before pruning
    ax1.plot(sizes, avg_time_before, '-', linewidth=linewidth, label='Before pruning', color=before_colour)
    ax1.fill_between(sizes, time_before_minus_std, time_before_plus_std, alpha=alpha, color=before_colour)

    # After pruning
    ax1.plot(sizes, avg_time_after, '-', linewidth=linewidth, label='After pruning', color=after_colour)
    ax1.fill_between(sizes, time_after_minus_std, time_after_plus_std, alpha=alpha, color=after_colour)

    # Labels
    ax1.set_ylabel('Runtime (sec)', fontsize=axis_font_size)
    ax1.legend(loc='upper left', prop={'size': legend_size})

    # Formatting
    ax1.margins(x=0)
    ax1.tick_params(axis='x', labelsize=tick_label_size)
    ax1.tick_params(axis='y', labelsize=tick_label_size)
    ax1.grid()

    # --- Percentage Pruning --- #
    # Nodes
    ax2.plot(sizes, avg_node_perc, '-', linewidth=linewidth, label='# Nodes', color=node_perc_colour)
    ax2.fill_between(sizes, node_perc_minus_std, node_perc_plus_std, alpha=alpha, color=node_perc_colour)

    # Edges
    ax2.plot(sizes, avg_edge_perc, '-', linewidth=linewidth, label='# Edges', color=edge_perc_colour)
    ax2.fill_between(sizes, edge_perc_minus_std, edge_perc_plus_std, alpha=alpha, color=edge_perc_colour)

    # Labels
    ax2.set_xlabel('Size (# Nodes)', fontsize=axis_font_size)
    ax2.set_ylabel('Reduction (%)', fontsize=axis_font_size)
    ax2.legend(loc='upper right', prop={'size': legend_size})

    # Formatting
    ax2.margins(x=0)
    ax2.tick_params(axis='x', labelsize=tick_label_size)
    ax2.tick_params(axis='y', labelsize=tick_label_size)
    ax2.grid()

    fig.set_size_inches(6, 7)

    # Use the window to manually adjust aspect ratios. The adjusted figure will
    # be saved once it is closed
    plt.show()
    fig.savefig('plots/results.pdf')
