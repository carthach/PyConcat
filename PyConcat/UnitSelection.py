import numpy as np
import HMM
import kBestViterbi.kBestViterbi as kb
import kBestViterbi.networkx_viterbi as kbg
from scipy.spatial import distance
from sklearn import preprocessing
from time import time

def fixDistanceMatrix(mat, type="min"):
    """Replace identical positions in the distance matrix with the max or min distance
       
    :param mat: the matrix to fix
     
    :param type: whether you want to replace all values by the min or max
     
    :return: the fixed distance matrix
    """

    if type == "min":
        mat[mat == 0.0] = np.inf
        mins = np.amin(mat, axis=1)
        mat[mat == np.inf] = mins
    else:
        mat[mat == 0.0] = np.inf
        # maxs = np.amax(mat, axis=1)
        # mat[mat == -np.inf] = maxs

    return mat

def computeDistanceMatrix(matrixA, matrixB):
    """Compute the distance matrix quickly with cdist
    
    :param matrixA: a 2D matrix
     
    :param matrixB: a 2D matrix
     
    :return: a distance matrix between matrixA and matrixB 
    """
    costMatrix = distance.cdist(matrixA, matrixB, 'euclidean')

    return costMatrix

def linearSearch(targetFeatures, corpusFeatures):
    """Brute force linear search, made a bit easier with python cdist to precompute matrices
    
    :param targetFeatures:
    
    :param corpusFeatures:
    
    :return: return the best sequence using brute force
    """
    targetCostMatrix = computeDistanceMatrix(targetFeatures, corpusFeatures)
    # concatenationCostMatrix = distance.cdist(corpusFeatures, corpusFeatures, 'euclidean')

    targetCostMatrixIndex = np.argsort(targetCostMatrix)

    sequence = targetCostMatrixIndex[:,0]

    #
    # for targetFeatureIndex, targetFeature in enumerate(targetFeatures[1:]):
    #     sequence.append(np.argmin(targetCostMatrix[targetFeatureIndex]))

    return sequence

def kdTree(targetFeatures, corpusFeatures):
    """ Faster than linearSearch
    
    :param targetFeatures:
    
    :param corpusFeatures:
    
    :return: the best sequence
    """

    from scipy import spatial

    tree = spatial.KDTree(corpusFeatures) #Frames
    a, b = tree.query(targetFeatures)

    return b

def viterbiOld(obs, states):
    """Modified version of Wikipedia Viterbi, adjusted for using costs
    
    :param obs: the target features
    
    :param states: the corpus features
    
    :return: the optimal state sequence
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

def normalise(array, method):
    """Normalise the arrays using Min/Max or Standard Deviation
    
    :param array: the array to normalise
     
    :param method: to normalise or standardise
     
    :return: the normalised array
    """
    scalar = None

    if method == "MinMax":
        scalar = preprocessing.MinMaxScaler()
    elif method == "SD":
        scalar = preprocessing.StandardScaler()

    return scalar.fit_transform(array)


def computeDistanceWithWeights(targetFeatures, corpusFeatures):
    """Perform distance computing with a different set of weights for the target and the corpus
    
    Need to figure out a more flexible way of doing this.
        
    :param targetFeatures:
     
    :param corpusFeatures:
     
    :return: the weighted distance matrices
    """
    energyWeight = 0.25
    mfccWeight = 0.25 / 12.0
    pitchWeight = 3.0

    targetFeatures = np.array(targetFeatures)
    corpusFeaturesWeighted = np.array(corpusFeatures)

    targetFeatures[:, 0] *= energyWeight
    targetFeatures[:, 1:-1] *= mfccWeight
    targetFeatures[:, -1] *= pitchWeight

    corpusFeaturesWeighted[:, 0] *= energyWeight
    corpusFeaturesWeighted[:, 1:-1] *= mfccWeight
    corpusFeaturesWeighted[:, -1] *= pitchWeight

    b = computeDistanceMatrix(targetFeatures, corpusFeaturesWeighted)

    energyWeight = 0.25
    mfccWeight = 2.0 / 12.0
    pitchWeight = 0.25

    corpusFeaturesWeighted = np.array(corpusFeatures)

    corpusFeaturesWeighted[:, 0] *= energyWeight
    corpusFeaturesWeighted[:, 1:-1] *= mfccWeight
    corpusFeaturesWeighted[:, -1] *= pitchWeight

    a = computeDistanceMatrix(corpusFeatures, corpusFeatures)

    return a, b

def unitSelection(targetFeatures, corpusFeatures, method="kdtree", normalise="MinMax", topK=30):
    """Optionally normalise and use one of the methods to return a sequence of indices
    
    :param targetFeatures:
    
    :param corpusFeatures:
    
    :param method: linearSearch, kdTree, viterbi, kViterbiExhaustive, kViterbiParallel, kViterbiGraph
    
    :param normalise: normalisation method
    
    :param topK: the number of paths to return (if using k-Best decoding)
    
    :return: the sequence path(s)
    """
    print "    Scaling and weighting feature vectors..."

    scalar = None

    if normalise == "MinMax":
        scalar = preprocessing.MinMaxScaler()
    elif normalise == "sd":
        scalar = preprocessing.StandardScaler()

    #If we need to perform scaling/normalisation
    if scalar:
        # targetFeatures = scalar.fit_transform(targetFeatures)
        # corpusFeatures = scalar.fit_transform(corpusFeatures)

        #Combine the two feature matrices
        combinedFeatures = np.concatenate((targetFeatures, corpusFeatures), axis=0)

        #Scale/normalise
        combinedFeatures = scalar.fit_transform(combinedFeatures)

        #Pop the two scaled matrices
        targetFeatures = combinedFeatures[:len(targetFeatures), :]
        corpusFeatures = combinedFeatures[len(targetFeatures):, :]

    #Call this method to compute the weighted a/b matrices for HMM
    print "    Computing distance matrices..."
    a, b = computeDistanceWithWeights(targetFeatures, corpusFeatures)

    print "    Performing unit selection..."
    targetCostWeight = 1.0
    concatCostWeight = 1.0

    if method is "kdTree":
        return kdTree(targetFeatures, corpusFeatures)
    elif method is "linearSearch":
        return linearSearch(targetFeatures, corpusFeatures)
    elif method is "viterbi":
        path, delta, phi, max_prob = kb.viterbiWithCosts(a, b, weights=(targetCostWeight, concatCostWeight))
        return path
    elif method is "kViterbiExhaustive":
        paths = kb.exhaustiveWithCosts(a, b)
        return paths
    elif method is "kViterbiParallel":
        paths, path_probs, delta, phi = kb.kViterbiParallelWithCosts(a, b, topK, weights=(targetCostWeight, concatCostWeight))
        return paths.tolist()
    elif method is "kViterbiGraph":
        a = computeDistanceMatrix(corpusFeatures, corpusFeatures)
        # a = fixDistanceMatrix(a, type="max")
        b = computeDistanceMatrix(targetFeatures, corpusFeatures)

        paths = kbg.kViterbiGraphWithCosts(a, b, topK, weights=(targetCostWeight, concatCostWeight))

        just_paths = [path[0] for path in paths]

        return just_paths
