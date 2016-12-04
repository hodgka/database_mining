import numpy as np
import argparse
import re
import networkx as nx
from collections import namedtuple

np.set_printoptions(threshold=np.inf, precision=4, suppress=True)


def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument('filename')
    args.add_argument('minsup', nargs='?')
    return args.parse_args()


def parse_data(fname):
    with open(fname, 'r') as f:
        graphs = []
        for line in f:
            # line = [t, #, num] or [v, id, label] or [e, id1, id2, label]
            line = line.strip().split()
            print("line: ", line)
            try:
                if line[0] == 'v':
                    graphs[-1].add_node(int(line[1]), label=line[2])
                elif line[0] == 'e':
                    graphs[-1].add_edge(int(line[1]), int(line[2]), label=line[3])
                else:
                    graphs.append(nx.Graph())
            except:
                print("Graph construction failed")
                pass
    return graphs


def gspan(C, D, minsup):
    E = right_most_path_extension(C, D)
    for t, sup_t in E:
        C_new = extend_C(t)
        C_new.sup_t = sup_t
        if sup_t >= minsup and is_canonical(C_new):
            gspan(C_new, D, minsup)


def right_most_path_extension(C, D):
    R = len(C.nodes())
    u_r = None  # TODO FIX THIS SHIT
    E = set(None)
    dfscode = namedtuple('DFScode', ['lx', 'ly', 'lxy'])
    for G_i in D:
        if nx.number_of_nodes(C) == 0:
            for set()
        else:
            isomorphisms = subgraph_ismorphisms(C, G_i)
            for phi in isomorphisms:
                for x in nx.neighbors(G_i, u_r)


# def subgraph_ismorphisms(C, G):
#     for graph in C:


if __name__ == "__main__":
    args = parse_arguments()
    graphs = parse_data(args.filename)
    thresh = int(args.minsup)
    print(graphs)
