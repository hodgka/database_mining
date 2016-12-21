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
                    graphs[-1].add_node(int(line[1]), dict(label=line[2], seen=False, id=))
                elif line[0] == 'e':
                    graphs[-1].add_edge(int(line[1]), int(line[2]), dict(label=line[3], direction=None))
                else:
                    graphs.append(nx.Graph())
            except:
                print("Graph construction failed")
                pass
    return graphs

def C_to_graph(C):
    G = nx.Graph()
    vertices = []
    vertex_labels = []
    edges = []
    edge_labels = []
    for u,v,l_u,l_v, l_uv in C:
        vertices.append(u)
        vertices.append(v)
        vertex_labels.append(l_u)
        vertex_labels.append(l_v)
        edges.append((u,v))
        edge_labels.append(l_uv)
    for i in range(len(vertices)):
        G.add_node(vertices[i], label=vertex_labels[i], id=i)
    for i in range(len(edges)):
        G.add_edge(edges[i], label=edge_labels[i])
    return G

def get_first_vertex(G):
    values = nx.get_node_attributes(G, 'id').items()
    values = [val[1] for val in values]
    return min(values)

def get_last_vertex(G):
    values = nx.get_node_attributes(G, 'id').items()
    values = [val[1] for val in values]
    return max(values)

def get_neighboring_edges(n):
    neighbors = G.neighbors(n)
    edges = [(n, node) for node in neighbors]


def dfs(G, n):



def gspan(C, D, minsup):
    E = right_most_path_extension(C, D)
    for t, sup_t in E:
        C_new = extend_C(t)
        C_new.sup_t = sup_t
        if sup_t >= minsup and is_canonical(C_new):
            gspan(C_new, D, minsup)


def right_most_path_extension(C, D):
    R = len(C.nodes())
    u_r = right_most_child(C)
    E = []
    for G_i in D:
        if len(C) == 0:
            for in G_i:
                f = (0, 1, )
        else:
            isomorphisms = subgraph_ismorphisms(C, G_i)
            for phi in isomorphisms:
                for x in nx.neighbors(G_i, u_r)


def subgraph_ismorphisms(C, G):
    phi = {}
    for t in C:
        u, v, l_u, l_v, l_uv = t
        phi_prime = []
        for partial in phi:
            if v > u:
                for x in G.neighbors(phi(u)):
                    if label(x) == label(v) and label(phi(u), x) = label(u, v):
                        phi_prime.extend(phi(v))
            else:
                if phi(v) in G.neighbors(phi(y)):
                    phi_prime.append(phi)
        phi = phi_prime
    return phi


def subgraph_ismorphisms(C, G):
    phi_c = []
    C_graph = C_to_graph(C)
    first_node = get_first_vertex(C_graph)
    for node in G.nodes():
        if G.node[node]['label'] == G.node[first_node]['label']:
            phi_c.append([(G.node[first_node]['id'], G.node[node]['id'])])
    for i, t in enumerate(C):
        u, v, l_u, l_v, l_uv = t
        phi_prime = []
        for phi in phi_c:
            if v > u:
                try:
                    phi_u = isomorphism(u, phi)
                except:
                    continue
                vertex = get_vertex(G, phi_u)
                neighbors = G.neighbors(vertex)
                neighboring_edges = get_neighboring_edges(vertex)
                for i in range(len(neighbors)):
                    inverse_isomorphism_exists = check_inverse_isomorphism(neighbors[i], phi)
                    if not inverse_isomorphism_exists and



if __name__ == "__main__":
    args = parse_arguments()
    graphs = parse_data(args.filename)
    thresh = int(args.minsup)
    print(graphs)
