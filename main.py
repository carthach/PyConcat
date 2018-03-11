#!/usr/local/bin/python
from PyConcat import Extractor
import PyConcat.UnitSelection as unitSelection
from PyConcat.Graphing import *

import os

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
    :param path: The path to the corpus
    :return: targetFile: The path to the target
    :return: corpusPath: The path to the corpus
    """
    files = os.listdir(path)

    targetFile = ""
    corpusPath = ""

    for file in files:
        fullFilePath = path + "/" + file
        if fullFilePath.endswith(('.mp3', '.wav')):
            targetFile = fullFilePath
        if fullFilePath.endswith("corpus"):
            corpusPath = fullFilePath
        # if os.path.isdir(fullFilePath):
        #     corpusPath = fullFilePath

    return targetFile, corpusPath

def parser(lgd=False, threshold=1.1):
    """
    Parses the command line arguments.
    :param lgd:       use local group delay weighting by default
    :param threshold: default value for threshold
    """
    import argparse
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    A tool to perform various concatenative synthesis techniques on the command line
    """)
    # general options
    p.add_argument('-v', dest='verbose', action='store_true',
                   help='be verbose')

    p.add_argument('-c', dest='config', action='store')

    p.add_argument('-timeScale', dest='timeScale', action='store', default='onsets',
                   help='The scale of the units: spectral, onsets and beats' ' [default=%(default)s]')
    p.add_argument('-writeOnsets', dest='writeOnsets', action='store_true', default=False,
                   help='Write out the segmented onsets' ' [default=%(default)s]')
    p.add_argument('-unitSelection', dest='unitSelection', action='store', default='linearSearch',
                   help='The unit selection method: linearSearch, kdTree, Viterbi' ' [default=%(default)s]')
    p.add_argument('-normalisation', dest='normalisation', action='store', default='MinMax',
                    help='Normalisation method: MinMax or SD' ' [default=%(default)s]')
    p.add_argument('-shouldStretch', dest='shouldStretch', action='store', default=False,
                   help='Stretch the units with Rubberband' ' [default=%(default)s]')

    p.add_argument('input', help='Input Data', action="store")
    p.add_argument('output', help='Output Path', action="store")

    args = p.parse_args()
    # print arguments
    if args.verbose:
        print(args)
    # return args

    #Override the command line arguments with YAML
    if args.config is not "":
        readYAMLConfig(args.config, args)

    return args

def main(args):
    """
    This shows how to input a folder for concatenative synthesis, segment/analyse then generate a sequence, write and plot
    :return:
    """
    import argparse

    extractor = Extractor.Extractor()

    #Settings
    timeScale = args.timeScale
    writeOnsets = args.writeOnsets
    unitSelectionMethod = args.unitSelection
    normalMethod = args.normalisation
    stretchUnits = args.stretchUnits

    #Create the output locations
    outputPath = args.output
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    #Extrapolate the target file and corpus folder
    targetFilename, corpusPath = getCorpus(args.input)

    #Get list of corpus files
    corpusFilenames = extractor.getListOfFiles(corpusPath, "*.mp3")

    print(corpusPath)

    #Segment and extract features
    print("Extracting Target")
    targetFeatures, targetUnits, targetUnitTimes = extractor.analyseFile(targetFilename, writeOnsets, scale=timeScale)
    print("Extracting Corpus")
    corpusFeatures, corpusUnits, corpusUnitTimes = extractor.analyseFiles(corpusFilenames, writeOnsets, scale=timeScale, yamlOutputFolder=outputPath)


    # # For graphing
    # costMatrix = computeDistanceMatrix(targetFeatures, targetFeatures)
    #
    # costMatrix = normalise(costMatrix, "MinMax")
    #
    # g = Graph.Adjacency(costMatrix.tolist())
    #
    # print costMatrix.toList()
    #
    # print(summary(g))
    #
    # layout = g.layout("kk")
    # plot(g, layout=layout, bbox=(300, 300), margin=20)
    #
    # # costMatrix = np.log(costMatrix)
    # createD3Diagram(costMatrix, outputPath)

    #Generate a sequence based on similarity
    print("Generating Sequence")

    sequence = unitSelection.unitSelection(targetFeatures, corpusFeatures, method=unitSelectionMethod, normalise=normalMethod, topK=10)

    #If it's spectral do IFFT resynthesis
    if timeScale is "spectral":
        audio = extractor.reSynth(sequence, corpusUnits)
    else:
        if isinstance(sequence, list): #If it's using kViterbi it returns a list (maybe should use some wildcard matching for more readability)
            for i, s in enumerate(sequence):
                audio = extractor.concatOnsets(s, corpusUnits, targetUnits, stretchUnits=stretchUnits)
                extractor.writeAudio(audio, outputPath + "/result_" + str(i) + ".wav")
        else:
            audio = extractor.concatOnsets(sequence, corpusUnits, targetUnits, stretchUnits=stretchUnits)
            extractor.writeAudio(audio, outputPath + "/result.wav")

    #Optionally plot data
    #plotData(sequence, targetFeatures, corpusFeatures)

    print("done")

def readYAMLConfig(filename, args):
    """
    Set the arguments in a class with settings from a file
    :param filename:
    :param args:
    :return:
    """
    import yaml

    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    #setattr Allows us to set variable class member names
    for section in cfg:
        setattr(args, section, cfg[section])
        # args[section] = cfg[section]

    return args

if __name__ == '__main__':
    # parse arguments
    args = parser()

    main(args)