#!/usr/bin/python
from Extractor import Extractor
from UnitSelection import *

import os

extractor = Extractor()

def plotData(sequence, targetFeatures, corpusFeatures):
    """
    Plot a comparison of the sequence versus the original target based on a feature
    :param sequence:
    :param targetFeatures:
    :param corpusFeatures:
    :return:
    """
    import matplotlib.pyplot as plt

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
    targetFeatures, targetUnits, targetUnitTimes = extractor.analyseFile(targetFilename, False, "onsets")
    print("Extracting Corpus")
    corpusFeatures, corpusUnits, corpusUnitTimes = extractor.analyseFiles(corpusFilenames, "onsets")

    #Generate a sequence based on similarity
    print("Generating Sequence")
    sequence = unitSelection(targetFeatures, corpusFeatures, method="viterbi")

    #If it's spectral-based used this
    # audio = extractor.reSynth(sequence, corpusFFTs)

    #If it's beats based use this
    audio = extractor.concatOnsets(sequence, corpusUnits)

    #Write out the audio
    extractor.writeAudio(audio, "/Users/carthach/Desktop/out.wav")

    #Optionally plot data
    #plotData(sequence, targetFeatures, corpusFeatures)

    print "done"

if __name__ == '__main__':
    # parse arguments
    main()