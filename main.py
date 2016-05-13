#!/usr/bin/python
from scipy.spatial import distance
from scipy import spatial
from Extractor import Extractor
import matplotlib.pyplot as plt
import numpy as np


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


def kdTree(targetFeatures, corpusFeatures):
    closestFileIndex = 0
    closestOnsetIndex = 0
    closestFrameIndex = 0


    tree = spatial.KDTree(corpusFeatures) #Frames
    a, b = tree.query(targetFeatures)

    return b

def linearSearchSimple(targetFeature, corpusFeatures, sequence):
    bestTargetCost = 0.0
    closestFileIndex = 0
    closestOnsetIndex = 0
    closestFrameIndex = 0

    start = True

    for fileIndex, corpusFile in enumerate(corpusFeatures):
        for onsetIndex, corpusOnset in enumerate(corpusFile):
            for frameIndex, corpusFrame in enumerate(corpusOnset):
                targetCost = distance.cdist([targetFeature], [corpusFrame], 'euclidean')

                if start:
                    bestTargetCost = targetCost
                    start = False
                elif targetCost < bestTargetCost:
                    bestTargetCost = targetCost
                    print("bestTargetCost: " + str(bestTargetCost))
                    closestFileIndex = fileIndex
                    closestOnsetIndex= onsetIndex
                    closestFrameIndex = frameIndex

    return closestFileIndex, closestOnsetIndex, closestFrameIndex

def unitSelection(targetFeature, corpusFeatures, sequence):
    bestTargetCost = 0.0
    bestConcatCost= 0.0
    bestTotalCost = 0.0
    closestFileIndex = 0
    closestOnsetIndex = 0
    closestOnsetTime = 0.0
    closestFrameIndex = 0

    targetCostWeight = 1.0
    concatCostWeight = 0.95

    start = True

    for fileIndex, corpusFile in enumerate(corpusFeatures):
        for onsetIndex, corpusOnset in enumerate(corpusFile):
            onsetTime = corpusOnset["onsetTime"]
            for frameIndex, corpusFrame in enumerate(corpusOnset["onsetFeatures"]):
                targetCost = distance.euclidean(targetFeature, corpusFrame["vector"])
                totalCost= targetCost

                concatCost = 0.0
                if len(sequence):
                    lastFileIndex, lastOnsetIndex, lastFrameIndex = sequence[-1]
                    lastFeatureVector = corpusFeatures[lastFileIndex][lastOnsetIndex]["onsetFeatures"][lastFrameIndex]["vector"]
                    concatCost = distance.euclidean(corpusFrame["vector"], lastFeatureVector)

                    #Compute total cost
                    totalCost = targetCostWeight * targetCost + concatCostWeight * concatCost

                if start:
                    bestTotalCost = totalCost
                    start = False
                elif totalCost < bestTotalCost:
                    bestTotalCost = totalCost

                    closestFileIndex = fileIndex
                    closestOnsetIndex= onsetIndex
                    closestFrameIndex = frameIndex

    return closestFileIndex, closestOnsetIndex, closestFrameIndex

def createSequence(targetFeatures, corpusFeatures):
    sequence = []

    # for targetOnsetCounter, targetOnset in enumerate(targetFeatures):
    #     print("Getting target onset "  + str(targetOnsetCounter) + " out of " + str(len(targetFeatures)))
    #     for targetFrameCounter, targetFrame in enumerate(targetOnset):
    #         print("Getting target frame " + str(targetFrameCounter) + " out of " + str(len(targetOnset)))
    #         # sequence.append(linearSearchSimple(targetFrame, corpusFeatures, sequence))
    #         sequence.append(kdTree(targetFrame, corpusFeatures))


    return kdTree(targetFeatures, corpusFeatures)



def writeSequence(sequence, corpusFilenames):

    fadeLength = 0
    rampUp = np.linspace(0.0, 1.0, fadeLength)
    rampDown = np.linspace(1.0, 0.0, fadeLength)


    audios = []
    for corpusFilename in corpusFilenames:
        audio = extractor.loadAudio(corpusFilename)
        audios.append(audio)

    outputAudio = []
    for unit in sequence:
        fileIndex, onsetTime, frameIndex = unit


        onsetTime = onsetTime * 44100
        frameTime = onsetTime + (frameIndex * 1024)
        endTime = int(frameTime+ 2048)
        if endTime > len(audios[fileIndex]):
            endTime = len(audios[fileIndex])

        audioSlice = audios[fileIndex][frameTime:endTime]

        #Fade out last n samples of outputAudio

        # if len(outputAudio):
        #     outputAudio[-fadeLength:] = outputAudio[-fadeLength:] * rampDown
        #     audioSlice[:fadeLength] = audioSlice[:fadeLength] * rampUp
        #
        #     outputAudio[-fadeLength:] = outputAudio[-fadeLength:] + audioSlice[:fadeLength]
        #     outputAudio = np.append(outputAudio, audioSlice[fadeLength:])
        # else:
        outputAudio = np.append(outputAudio, audioSlice)

    extractor.writeAudio(outputAudio, "/Users/carthach/Desktop/out.wav")

def main():
    """

    :return:
    """


    # targetFilename = "/Users/carthach/Desktop/debug_audio/breaks/5th Dimension - Rainmaker (part2).wav"
    # targetFilename = "/Users/carthach/Desktop/debug_audio/beatport/0297579 Kelly Holiday - Moscow Time (Electro Mix) [Doctormusik Records] == Electro House === Am.mp3"
    # corpusFilenames = extractor.getListOfWavFiles("/Users/carthach/Desktop/debug_audio/breaks")

    targetFilename = "/Users/carthach/Desktop/debug_audio/python_test/melody9.wav"
    corpusFilenames = extractor.getListOfWavFiles("/Users/carthach/Desktop/debug_audio/python_test/corpus")

    # a = extractor.loadAudio(targetFilename)
    # b = extractor.synthResynth(a)
    # extractor.writeAudio(b, "/Users/carthach/Desktop/out.wav")

    # Concatenative Synthesis
    targetFeatures, targetFFTs = extractor.analyseFile(targetFilename, False, False)
    corpusFeatures, corpusFFTs  = extractor.analyseFiles(corpusFilenames)

    sequence = createSequence(targetFeatures, corpusFeatures)
    audio = extractor.resynthesise_audio(sequence, corpusFFTs)
    extractor.writeAudio(audio, "/Users/carthach/Desktop/out.wav")

    plotData(sequence, targetFeatures, corpusFeatures)

    #Test segmenter
    # audio = extractor.loadAudio(targetFilename)
    # onsetTimes = extractor.extractOnsets(targetFilename)[0]
    # segments = extractor.segmentSignal(onsetTimes, audio)

    print "done"

if __name__ == '__main__':
    # parse arguments
    main()