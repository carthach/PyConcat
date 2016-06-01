import numpy as np
from scipy.spatial import distance

def linearSearch(targetFeatures, corpusFeatures):
    """
    Brute force linear search, made a bit easier with python cdist to precompute matrices
    :param targetFeatures:
    :param corpusFeatures:
    :return:
    """


    targetCostMatrix = distance.cdist(targetFeatures, corpusFeatures, 'euclidean')
    concatenationCostMatrix = distance.cdist(corpusFeatures, corpusFeatures, 'euclidean')

    for targetFeatureIndex, targetFeature in enumerate(targetFeatures[1:]):
        pass

    return 0

def kdTree(targetFeatures, corpusFeatures):
    """
    Faster than linearSearch
    :param targetFeatures:
    :param corpusFeatures:
    :return:
    """

    from scipy import spatial

    tree = spatial.KDTree(corpusFeatures) #Frames
    a, b = tree.query(targetFeatures)

    return b

def viterbi(obs, states):
    """
    Modified version of wikipedia viterbi
    :param obs:
    :param states:
    :return:
    """
    trans_p = distance.cdist(states, states, 'euclidean')
    trans_p[trans_p == 0.0] = np.inf
    emit_p = distance.cdist(obs, states, 'euclidean')

    V = [{}]
    path = {}

    for y in range(len(states)):
        V[0][y] = emit_p[0][y]
        path[y] = y

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append([])
        newpath = {}

        for yIndex, y in enumerate(trans_p):
            costs = [V[t - 1][y0Index] + trans_p[yIndex][y0Index] + emit_p[t][yIndex] for y0Index, y0 in enumerate(trans_p[yIndex])]

            minCost = np.amin(costs, axis=0)
            minIndex = np.argmin(costs, axis=0)

            V[t].append(minCost)

            newpath[yIndex] = np.append(path[minIndex], [yIndex])

        # Don't need to remember the old paths
        path = newpath

    n = 0  # if only one element is observed max is sought in the initialization values
    if len(obs) != 1:
        n = t

    (prob, state) = min((V[n][yIndex], yIndex) for yIndex, y in enumerate(trans_p))

    minCost = np.amin(costs, axis=0)
    minIndex = np.argmin(costs, axis=0)

    return path[state]

def unitSelection(targetFeatures, corpusFeatures, method="kdtree", normalise=True):
    """
    Optionally normalise and use one of the methods to return a sequence of indices
    :param targetFeatures:
    :param corpusFeatures:
    :param method:
    :param normalise:
    :return:
    """

    from sklearn import preprocessing

    if normalise:
        min_max_scaler = preprocessing.MinMaxScaler()
        targetFeatures = min_max_scaler.fit_transform(targetFeatures)
        corpusFeatures = min_max_scaler.fit_transform(corpusFeatures)

    if method is "kdTree":
        return kdTree(targetFeatures, corpusFeatures)
    elif method is "linearSearch":
        return linearSearch(targetFeatures, corpusFeatures)
    elif method is "viterbi":
        return viterbi(targetFeatures, corpusFeatures)