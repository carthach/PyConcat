import numpy as np
import MyHMM as hmm
from scipy.spatial import distance

def linearSearch(targetFeatures, corpusFeatures):
    """
    Brute force linear search, made a bit easier with python cdist to precompute matrices
    :param targetFeatures:
    :param corpusFeatures:
    :return:
    """
    targetCostMatrix = distance.cdist(targetFeatures, corpusFeatures, 'euclidean')
    # concatenationCostMatrix = distance.cdist(corpusFeatures, corpusFeatures, 'euclidean')

    sequence = []

    for targetFeatureIndex, targetFeature in enumerate(targetFeatures[1:]):
        sequence.append(np.argmin(targetCostMatrix[targetFeatureIndex]))

    return sequence

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

def viterbiOld(obs, states):
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

def viterbi(self, x):
    ''' Given sequence of emissions, return the most probable path
        along with its probability. '''
    x = map(self.smap.get, x)  # turn emission characters into ids
    nrow, ncol = len(self.Q), len(x)
    mat = numpy.zeros(shape=(nrow, ncol), dtype=float)  # prob
    matTb = numpy.zeros(shape=(nrow, ncol), dtype=int)  # backtrace
    # Fill in first column
    for i in xrange(0, nrow):
        mat[i, 0] = self.E[i, x[0]] * self.I[i]
    # Fill in rest of prob and Tb tables
    for j in xrange(1, ncol):
        for i in xrange(0, nrow):
            ep = self.E[i, x[j]]
            mx, mxi = mat[0, j - 1] * self.A[0, i] * ep, 0
            for i2 in xrange(1, nrow):
                pr = mat[i2, j - 1] * self.A[i2, i] * ep
                if pr > mx:
                    mx, mxi = pr, i2
            mat[i, j], matTb[i, j] = mx, mxi
    # Find final state with maximal probability
    omx, omxi = mat[0, ncol - 1], 0
    for i in xrange(1, nrow):
        if mat[i, ncol - 1] > omx:
            omx, omxi = mat[i, ncol - 1], i
    # Backtrace
    i, p = omxi, [omxi]
    for j in xrange(ncol - 1, 0, -1):
        i = matTb[i, j]
        p.append(i)
    p = ''.join(map(lambda x: self.Q[x], p[::-1]))
    return omx, p  # Return probability and path

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
    elif method is "Markov":
        # return viterbi(targetFeatures, corpusFeatures)
        myHMM = hmm.MyHMM(targetFeatures, corpusFeatures)

        return myHMM.viterbi()
