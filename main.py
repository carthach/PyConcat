#!/usr/bin/python
from scipy.spatial import distance
from Extractor import Extractor
import numpy as np

extractor = Extractor()

def closestUnit(targetFeature, corpusFeatures, sequence):
    targetCost = 0.0
    concatCost = 0.0
    totalCost = 0.0
    closestFileIndex = 0
    closestOnsetIndex = 0
    closestOnsetTime = 0.0
    closestFrameIndex = 0

    start = True

    for fileIndex, corpusFile in enumerate(corpusFeatures):
        for onsetIndex, corpusOnset in enumerate(corpusFile):
            onsetTime = corpusOnset["onsetTime"]
            for frameIndex, corpusFrame in enumerate(corpusOnset["onsetFeatures"]):
                currentTargetCost = distance.euclidean(targetFeature, corpusFrame["vector"])
                currentCost = currentTargetCost

                # currentConcatCost = 0.0
                # if len(sequence):
                #     lastFileIndex, lastOnsetIndex, lastFrameIndex = sequence[-1]
                #     lastFeatureVector = corpusFeatures[lastFileIndex][lastOnsetIndex]["onsetFeatures"][lastFrameIndex]["vector"]
                #     currentConcatCost = distance.euclidean(corpusFrame["vector"], lastFeatureVector)
                #
                #     currentCost = currentCost + currentConcatCost * 0.5

                if start:
                    totalCost = currentCost
                    start = False
                elif currentCost < totalCost:
                    totalCost = currentCost
                    closestFileIndex = fileIndex
                    closestOnsetIndex= onsetIndex
                    closestFrameIndex = frameIndex

    return closestFileIndex, closestOnsetIndex, closestFrameIndex

def createSequence(targetFeatures, corpusFeatures):
    sequence = []

    for targetOnset in targetFeatures:
        print("Getting target onset")
        for targetFrame in targetOnset["onsetFeatures"]:
            sequence.append(closestUnit(targetFrame["vector"], corpusFeatures, sequence))

    return sequence

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

    # #Analyse the file and keep the features but not the audio
    targetFeatures = extractor.analyseFile(targetFilename, False)
    corpusFeatures = extractor.analyseFiles(corpusFilenames)
    sequence = createSequence(targetFeatures, corpusFeatures)
    audio = extractor.resynthesise_audio(sequence, corpusFeatures)
    extractor.writeAudio(audio, "/Users/carthach/Desktop/out.wav")

    # extractor.writeAudio(extractor.resynthesise_audio(sequence, corpusFeatures), "/Users/carthach/Desktop/out.wav")
    print("Done")
    # onsets = extractor.extractOnsets(targetFilename)[0]

    # extractor.writeOnsets(onsets, targetFilename)
    # features = extractor.extractFeatures(onsets[0])
    # audio = extractor.loadAudio(targetFilename)
    # extractor.testing(onsets[13], audio)

    #corpusFeatures = extractor.analyseFiles(corpusFiles)

    #sequence = createSequence(targetFeatures, corpusFeatures)
    #writeSequence(sequence, "out.wav")

if __name__ == '__main__':
    # parse arguments
    main()