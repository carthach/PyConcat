#!/usr/local/bin/python
from PyConcat import Extractor
import PyConcat.UnitSelection as unitSelection
from PyConcat.Graphing import *

import os

extractor = Extractor.Extractor()

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

    # p.add_argument('-i', '--input', help='Input file name', required=True)
    # p.add_argument('-o', '--output', help='Output file name', required=True)

    p.add_argument('-scale', dest='scale', action='store', default='scale',
                   help='The scale of the units: spectral, onsets and beats' ' [default=%(default)s]')
    p.add_argument('-writeOnsets', dest='writeOnsets', action='store_true', default=False,
                   help='Write out the segmented onsets' ' [default=%(default)s]')
    p.add_argument('-method', action='store', default='linearSearch',
                   help='The unit selection method: linearSearch, kdTree, Viterbi' ' [default=%(default)s]')
    p.add_argument('-norm', action='store', default='MinMax',
                    help='Normalisation method: MinMax or SD' ' [default=%(default)s]')

    p.add_argument('input', help='Input Data', action="store")
    p.add_argument('output', help='Output Path', action="store")



    # online / offline mode
    # # wav options
    # wav = p.add_argument_group('audio arguments')
    # wav.add_argument('--norm', action='store_true', default=None,
    #                  help='normalize the audio (switches to offline mode)')
    # wav.add_argument('--att', action='store', type=float, default=None,
    #                  help='attenuate the audio by ATT dB')
    # # spectrogram options
    # spec = p.add_argument_group('spectrogram arguments')
    # spec.add_argument('--fps', action='store', default=200, type=int,
    #                   help='frames per second [default=%(default)s]')
    # spec.add_argument('--frame_size', action='store', type=int, default=2048,
    #                   help='frame size [samples, default=%(default)s]')
    # spec.add_argument('--ratio', action='store', type=float, default=0.5,
    #                   help='window magnitude ratio to calc number of diff '
    #                        'frames [default=%(default)s]')
    # spec.add_argument('--diff_frames', action='store', type=int, default=None,
    #                   help='diff frames')
    # spec.add_argument('--max_bins', action='store', type=int, default=3,
    #                   help='bins used for maximum filtering '
    #                        '[default=%(default)s]')
    # # LGD stuff
    # mask = p.add_argument_group('local group delay based weighting')
    # mask.add_argument('--lgd', action='store_true', default=lgd,
    #                   help='apply local group delay based weighting '
    #                        '[default=%(default)s]')
    # mask.add_argument('--temporal_filter', action='store', default=3, type=int,
    #                   help='apply a temporal filter of N frames before '
    #                        'calculating the LGD weighting mask '
    #                        '[default=%(default)s]')
    # # filtering
    # filt = p.add_argument_group('magnitude spectrogram filtering arguments')
    # filt.add_argument('--no_filter', dest='filter', action='store_false',
    #                   default=True, help='do not filter the magnitude '
    #                                      'spectrogram with a filterbank')
    # filt.add_argument('--fmin', action='store', default=30, type=float,
    #                   help='minimum frequency of filter '
    #                        '[Hz, default=%(default)s]')
    # filt.add_argument('--fmax', action='store', default=17000, type=float,
    #                   help='maximum frequency of filter '
    #                        '[Hz, default=%(default)s]')
    # filt.add_argument('--bands', action='store', type=int, default=24,
    #                   help='number of bands per octave [default=%(default)s]')
    # filt.add_argument('--equal', action='store_true', default=False,
    #                   help='equalize triangular windows to have equal area')
    # filt.add_argument('--block_size', action='store', default=2048, type=int,
    #                   help='perform filtering in blocks of N frames '
    #                        '[default=%(default)s]')
    # # logarithm
    # log = p.add_argument_group('logarithmic magnitude spectrogram arguments')
    # log.add_argument('--no_log', dest='log', action='store_false',
    #                  default=True, help='use linear magnitude scale')
    # log.add_argument('--mul', action='store', default=1, type=float,
    #                  help='multiplier (before taking the log) '
    #                       '[default=%(default)s]')
    # log.add_argument('--add', action='store', default=1, type=float,
    #                  help='value added (before taking the log) '
    #                       '[default=%(default)s]')
    # # onset detection
    # onset = p.add_argument_group('onset peak-picking arguments')
    # onset.add_argument('-t', dest='threshold', action='store', type=float,
    #                    default=threshold, help='detection threshold '
    #                                            '[default=%(default)s]')
    # onset.add_argument('--combine', action='store', type=float, default=0.03,
    #                    help='combine onsets within N seconds '
    #                         '[default=%(default)s]')
    # onset.add_argument('--pre_avg', action='store', type=float, default=0.15,
    #                    help='build average over N previous seconds '
    #                         '[default=%(default)s]')
    # onset.add_argument('--pre_max', action='store', type=float, default=0.01,
    #                    help='search maximum over N previous seconds '
    #                         '[default=%(default)s]')
    # onset.add_argument('--post_avg', action='store', type=float, default=0,
    #                    help='build average over N following seconds '
    #                         '[default=%(default)s]')
    # onset.add_argument('--post_max', action='store', type=float, default=0.05,
    #                    help='search maximum over N following seconds '
    #                         '[default=%(default)s]')
    # onset.add_argument('--delay', action='store', type=float, default=0,
    #                    help='report the onsets N seconds delayed '
    #                         '[default=%(default)s]')
    # version
    # p.add_argument('--version', action='version',
    #                version='%(prog)spec 1.03 (2014-11-02)')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return args
    return args

def main(args):
    """
    This shows how to input a folder for concatenative synthesis, segment/analyse then generate a sequence, write and plot
    :return:
    """
    import argparse

    #Settings
    scale = args.scale
    writeOnsets = args.writeOnsets
    unitSelectionMethod = args.method
    normalMethod = args.norm

    #Create the output locations
    outputPath = args.output
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    #Extrapolate the target file and corpus folder
    targetFilename, corpusPath = getCorpus(args.input)

    #Get list of corpus files
    corpusFilenames = extractor.getListOfWavFiles(corpusPath)

    #Segment and extract features
    print("Extracting Target")
    targetFeatures, targetUnits, targetUnitTimes = extractor.analyseFile(targetFilename, writeOnsets, scale)
    print("Extracting Corpus")
    corpusFeatures, corpusUnits, corpusUnitTimes = extractor.analyseFiles(corpusFilenames, writeOnsets, scale=None)

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

    sequence = unitSelection.unitSelection(targetFeatures, corpusFeatures, method="kViterbiParallel", normalise=normalMethod, topK=10)

    #If it's spectral do IFFT resynthesis
    if scale is "spectral":
        audio = extractor.reSynth(sequence, corpusUnits)
    else:
        if isinstance(sequence, list): #If it's using kViterbi it returns a list (maybe should use some wildcard matching for more readability)
            for i, s in enumerate(sequence):
                audio = extractor.concatOnsets(s, corpusUnits, targetUnits, shouldStretch=True)
                extractor.writeAudio(audio, outputPath + "/result_" + str(i) + ".wav")
        else:
            audio = extractor.concatOnsets(sequence, corpusUnits, targetUnits, shouldStretch=True)
            extractor.writeAudio(audio, outputPath + "/result.wav")

    #Optionally plot data
    #plotData(sequence, targetFeatures, corpusFeatures)

    print "done"

#Debug method when not using from the command line
def debugArgs():
    class Args:
        pass

    args = Args()

    args.scale = "onsets" #spectral, onsets, beat or None to use the whole file
    args.writeOnsets = False
    args.method = "kdTree" #Options are "linearSearch, kdTree, viterbi, kViterbiParallel, kViterbiGraph"
    args.norm =  "SD"


    # args.input = "/Users/carthach/Desktop/debug_audio/breaks_pyconcat"
    # args.input = "/Users/carthach/Desktop/debug_audio/beatport_test2"
    args.input = "/Users/carthach/Desktop/debug_audio/scale_test"
    # args.input = "/Users/carthach/Google Drive/Tmp/test_stuff/scale_test"

    args.output = "/Users/carthach/Desktop/concat_out"

    return args

if __name__ == '__main__':
    # parse arguments
    # args = parser()

    #If you're not running from the command line use this and set in the method
    args = debugArgs()

    main(args)