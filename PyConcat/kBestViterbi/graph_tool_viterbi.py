from graph_tool.all import *
import numpy as np

# Create a NetworkX compatible directed graph that represents a HMM
def createViterbiGraphWithCosts(a, b, weights=(1.0, 1.0)):
    """Return a NetworkX compabitible graph for computing k Best Paths for Viterbi Decoding

    Uses costs for concatenative synthesis purposes

    :param a: 2d numpy transition matrix

    :param b: 2d numpy emission matrix

    :param topK: the number of paths we want to retain

    :param weights: the target cost weighting and concatenation cost weighting

    :return: a Directed Acyclic Graph representing a HMM
    """
    nObs = len(b)
    nStates = len(a)

    # Number of nodes in the graph
    nNodes = nObs * nStates

    # Create a directed graph
    G = Graph()
    edgeWeightDict = G.new_edge_property("float")

    G.edge_properties["weight"] = edgeWeightDict

    # Create all the necessary nodes
    for n in range(0, nNodes):
        G.add_vertex(n)

    # Create the start and dummy end nodes
    startVertexIndex = G.add_vertex()
    endVertexIndex = G.add_vertex()

    targetCostsTotalWeight = weights[0]
    concatCostsTotalWeight = weights[1]

    # Add the weights for the start node
    for i in range(nStates):
        w = b[0, i]
        edgeIndex = G.add_edge(startVertexIndex, i)
        edgeWeightDict[edgeIndex] = w

        edgeIndex = G.add_edge((nNodes - 1) - i, endVertexIndex)
        edgeWeightDict[edgeIndex] = 1

    for t in range(0, nNodes - nStates, nStates):
        i_offset = t
        j_offset = t + nStates

        real_t = t / nStates + 1

        for i in range(nStates):
            i_idx = i_offset + i

            for j in range(nStates):
                j_idx = j_offset + j

                w = (targetCostsTotalWeight * b[real_t, j]) + (concatCostsTotalWeight * a[i, j])

                edgeIndex = G.add_edge(i_idx, j_idx)

    # This for the mod operation in the shortest path computation
    graphProperty = G.new_graph_property("int")
    G.graph_properties["nStates"] = graphProperty
    G.graph_properties["nStates"] = nStates

    return G, startVertexIndex, endVertexIndex


def shortestPaths(G, startVertexIndex, endVertexIndex, topK, negativeLogSpace=True):
    """Compute the k Shortest Paths, optionally in negative log space

    :param G: A directed acyclic graph

    :param topK: the number pof paths

    :param negativeLogSpace: use negative log space

    :return: The paths and their costs
    """
    if negativeLogSpace:
        for e in G.edges():
            edgeWeight = g.ep.weight[e]

            edgeWeight = np.log(edgeWeight)
            edgeWeight = - edgeWeight

            g.ep.weight[e]


    # def k_shortest_paths(G, source, target, k, weight=None):
    #     from itertools import islice
    #     return list(islice(shortest_simple_paths_with_costs(G, source, target, weight=weight, topK=k), k))

    # Some params
    # source = -1
    # target = len(G) - 2
    nStates = G.graph_properties["nStates"]

    # pathsAndCosts = k_shortest_paths(G, source, target, topK, weight="weight")

    pathsAndCosts = all_shortest_paths(G, startVertexIndex, endVertexIndex,G.edge_properties["weight"])

    # pathsAndCosts = k_shortest_paths(G, startVertexIndex, endVertexIndex, topK, weight="weight")

    # Antilog and negate to get the correct probabilities
    # if negativeLogSpace:
    #     pathsAndCosts = [(p[0], np.exp(-p[1])) for p in pathsAndCosts]

    # Do a mod by number of states, and remove the dummy nodes
    # pathsAndCosts = [(np.mod(p[0][1:-1], nStates), p[1]) for p in pathsAndCosts]

    return pathsAndCosts

def kViterbiGraphWithCosts(a, b, topK, weights=(1.0, 1.0)):
    """Compute k Best paths using k shortest paths decoding

    Uses weighted costs for concatenative synthesis purposes

    :param a: the transition matrix

    :param b: the emission matrix

    :param topK: the number of paths to return

    :param weights: target and concatenation weights

    :return: the paths and their costs
    """
    G, startVertexIndex, endVertexIndex = createViterbiGraphWithCosts(a, b, weights=weights)
    # G = createPrunedViterbiGraphWithCosts(a, b, topK, weights=weights)

    paths = shortestPaths(G, startVertexIndex, endVertexIndex, topK, negativeLogSpace=False)

    return paths