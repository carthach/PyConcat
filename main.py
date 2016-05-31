#!/usr/bin/python
from scipy.spatial import distance
from scipy import spatial
from Extractor import Extractor
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import os

extractor = Extractor()

def plotData(sequence, targetFeatures, corpusFeatures):
    targetPoints = []
    corpusPoints = []

    for onset in targetFeatures:
        for frame in onset:
            targetPoints.append(frame)

    for unit in sequence:
        fileIndex, onsetIndex, frameIndex = unit

        corpusPoints.append(corpusFeatures[fileIndex][onsetIndex][frameIndex])

    plt.subplot(211)
    plt.plot(targetPoints)
    plt.subplot(212)
    plt.plot(corpusPoints)

    plt.show()

def viterbi(obs, states):
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
            costs = [V[t - 1][y0Index] + trans_p[y0Index][yIndex] + emit_p[t][yIndex] for y0Index, y0 in enumerate(trans_p[yIndex])]

            minCost = np.amin(costs, axis=0)
            minIndex = np.argmin(costs, axis=0)

            V[t].append(minCost)

            newpath[y] = path[minIndex] + [y]

            print y

        print t
        # Don't need to remember the old paths
        path = newpath

    n = 0  # if only one element is observed max is sought in the initialization values
    if len(obs) != 1:
        n = t

    (prob, state) = min((V[n][y], y) for y in emit_p)
    return (prob, path[state])

def kdTree(targetFeatures, corpusFeatures):
    """
    Faster than linearSearch
    :param targetFeatures:
    :param corpusFeatures:
    :return:
    """
    tree = spatial.KDTree(corpusFeatures) #Frames
    a, b = tree.query(targetFeatures)

    return b

def linearSearch(targetFeatures, corpusFeatures):
    from scipy.spatial import distance


    targetCostMatrix = distance.cdist(targetFeatures, corpusFeatures, 'euclidean')
    concatenationCostMatrix = distance.cdist(corpusFeatures, corpusFeatures, 'euclidean')

    for targetFeatureIndex, targetFeature in enumerate(targetFeatures[1:]):
        pass




    return 0






def unitSelection(targetFeatures, corpusFeatures, method="kdtree", normalise=True):
    """
    Optionally normalise and use one of the methods to return a sequence of indices
    :param targetFeatures:
    :param corpusFeatures:
    :param method:
    :param normalise:
    :return:
    """
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



def getCorpus(path):
    """
    Utility tool to return the target and corpus path for a given path
    :param path:
    :return:
    """
    files = os.listdir(path)

    targetFile = ""
    corpusPath = ""

    for file in files:
        fullFilePath = path + "/" + file
        if fullFilePath.endswith(('.mp3', '.wav')):
            targetFile = fullFilePath
        if os.path.isdir(fullFilePath):
            corpusPath = fullFilePath

    return targetFile, corpusPath


def main():
    """
    This shows how to input a folder for concatenative synthesis, segment/analyse then generate a sequence, write and plot
    :return:
    """

    #Extrapolate the target file and corpus folder and get the list of corpus files
    targetFilename, corpusPath = getCorpus("/Users/carthach/Desktop/debug_audio/python_test")
    # targetFilename, corpusPath = getCorpus("/Users/carthach/Desktop/debug_audio/beatport_test")

    corpusFilenames = extractor.getListOfWavFiles(corpusPath)

    #Segment and extract features
    print("Extracting Target")
    targetFeatures, targetUnits, targetUnitTimes = extractor.analyseFile(targetFilename, False, "beats")
    print("Extracting Corpus")
    corpusFeatures, corpusUnits, corpusUnitTimes = extractor.analyseFiles(corpusFilenames, "beats")

    #Generate a sequence based on similarity
    print("Generating Sequence")
    sequence = unitSelection(targetFeatures, corpusFeatures, method="viterbi")

    #If it's spectral-based used this
    # audio = extractor.reSynth(sequence, corpusFFTs)

    #If it's beats based use this
    audio = extractor.concatBeats(sequence, corpusUnits, targetUnitTimes)

    #Write out the audio
    extractor.writeAudio(audio, "/Users/carthach/Desktop/out.wav")

    #Optionally plot data
    #plotData(sequence, targetFeatures, corpusFeatures)

    print "done"

if __name__ == '__main__':
    # parse arguments
    main()