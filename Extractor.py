from __future__ import print_function
import essentia
import essentia.standard
import numpy as np
import matplotlib.pyplot as plt
import peakutils

class Extractor:
    #Features
    loader = None
    writer = None
    onsetRate = None
    slicer = None
    duration = None
    mfcc = None
    fft = None
    magnitude = None
    spectrum = None
    w = None
    pitch = None

    frameSize = 2048
    hopSize = 0

    def synthResynth(self, audio):
        fft = essentia.standard.FFT()
        ifft = essentia.standard.IFFT()
        overlapAdd = essentia.standard.OverlapAdd(frameSize = self.frameSize, hopSize = self.hopSize)

        audio_out = []
        for fstart in range(0, len(audio) - self.frameSize, self.hopSize):
            frame = audio[fstart:fstart + self.frameSize]
            fft_frame = fft(self.w(frame))

            ifft_frame = ifft(fft_frame)
            # ifft_frame = overlapAdd(ifft_frame)

            audio_out = np.append(audio_out, ifft_frame)

        return audio_out




        return audio

    def resynthesise_audio(self, sequence, features):
        audio = []
        ifft = essentia.standard.IFFT()
        overlap = essentia.standard.OverlapAdd(frameSize = self.frameSize, hopSize = self.hopSize)

        for unit in sequence:
            fileIndex, onsetIndex, frameIndex = unit

            clip = ifft(features[fileIndex][onsetIndex]["onsetFeatures"][frameIndex]["fft"])
            # clip = self.w(clip)
            clip = overlap(clip)


            audio = np.append(audio, clip)

        return audio

    def correlation(self,audio, main_audio):

        cc = essentia.standard.CrossCorrelation(maxLag=len(main_audio))

        cc_out = cc(main_audio, audio)

        plt.subplot(311)
        plt.plot(main_audio)
        plt.subplot(312)
        plt.plot(audio)
        plt.subplot(313)
        plt.plot(cc_out)

        plt.show()

    def testing(self, audio, main_audio):

        cc = essentia.standard.AutoCorrelation()

        cc_out = cc(main_audio)

        peaks = peakutils.indexes(cc_out, 0.8, min_dist=50)

        plt.subplot(311)
        plt.plot(main_audio)
        plt.subplot(312)
        plt.plot(cc_out)
        plt.plot(peaks, 'ro')

        plt.show()


    def extractFeatures(self,audio):
        """
        Extract features from an audio vector
        :param audio:
        :return:
        """

        pool = essentia.Pool()

        features = []

        for fstart in range(0, len(audio) - self.frameSize, self.hopSize):
            frame = audio[fstart:fstart + self.frameSize]

            fft_frame = self.fft(self.w(frame))
            mag = self.magnitude(fft_frame)

            pitch, pitchConfidence = self.pitch(mag)

            mfcc_bands, mfcc_coeffs = self.mfcc(mag)

            frameFeatures = {}

            frameFeatures["vector"] = pitch
            # frameFeatures["vector"] = np.append(frameFeatures["vector"], mfcc_coeffs)
            frameFeatures["fft"] = fft_frame

            features.append(frameFeatures)

        self.pitch.reset()
        self.mfcc.reset()

        # for frame in essentia.standard.FrameGenerator(audio, frameSize=1024, hopSize=512):
        #     mfcc_bands, mfcc_coeffs = self.mfcc(self.spectrum(self.w(frame)))
        #     pool.add('lowlevel.mfcc', mfcc_coeffs)
        #     pool.add('lowlevel.mfcc_bands', mfcc_bands)

        return features

    def loadAudio(self, fileName):

        audio = None

        if fileName:
            self.loader.configure(filename=fileName)

            # and then we actually perform the loading:
            audio = self.loader()

        return audio


    def extractOnsets(self,fileName):
        """
        Extract and return a vector of onsets as audio vectors
        :param file:
        :return:
        """

        slices = None
        onsetTimes = None

        if fileName:
            audio = self.loadAudio(fileName)
            onsetRateResult = self.onsetRate(audio)
            self.onsetRate.reset()

            onsetTimes, onsetRate  = onsetRateResult
            endTimes = onsetTimes[1:]
            d = self.duration(audio)
            endTimes = np.append(endTimes, d)
            endTimes = essentia.array(endTimes)

            self.slicer.configure(startTimes=onsetTimes, endTimes=endTimes)
            slices = self.slicer(audio)
            self.slicer.reset()

        return onsetTimes, slices, fileName

    def writeAudio(self, audio, fileName):
        self.writer = essentia.standard.MonoWriter()
        self.writer.configure(filename=fileName)
        audio = essentia.array(audio)
        self.writer(audio)

    def writeOnsets(self, onsets, fileName):
        import os

        i = 0
        fileNameOnly, fileNameExt = os.path.splitext(fileName)

        fileNames = []

        for onset in onsets:
            onsetFilename = fileNameOnly + "_" + str(i) + fileNameExt

            self.writeAudio(onset, onsetFilename)

            fileNames.append(onsetFilename)
            i = i + 1

        return fileNames

    def analyseAudio(self, audio, writeOnsets):
        """
        Extract onsets from a single audio then extract features from all those onsets
        :param file:
        :return:
        """
        onsetTimes, onsets, fileName = self.extractOnsets(file)

        if (writeOnsets):
            fileNames = self.writeOnsets(onsets, file)

        fileFeatures = []

        for onsetTime, onset in zip(onsetTimes, onsets):
            features = self.extractFeatures(onset)

            onsetFeatures = {}
            onsetFeatures["onsetTime"] = onsetTime
            onsetFeatures["onsetFeatures"] = features

            fileFeatures.append(onsetFeatures)

        return fileFeatures

    def analyseFile(self,file, writeOnsets):
        """
        Extract onsets from a single file then extract features from all those onsets
        :param file:
        :return:
        """
        onsetTimes, onsets, fileName = self.extractOnsets(file)

        print("Processing file: " + file)

        if (writeOnsets):
            fileNames = self.writeOnsets(onsets, file)

        fileFeatures = []

        for onsetTime, onset in zip(onsetTimes, onsets):
            features = self.extractFeatures(onset)

            onsetFeatures = {}
            onsetFeatures["onsetTime"] = onsetTime
            onsetFeatures["onsetFeatures"] = features

            fileFeatures.append(onsetFeatures)

        return fileFeatures

    def analyseFiles(self,listOfFiles):
        """
        Perform onset detection and extract features from all the onsets from all the files
        :param listOfFiles:
        :return:
        """
        filesFeatures = []

        for file in listOfFiles:
            fileFeatures = self.analyseFile(file, False)
            filesFeatures.append(fileFeatures)

        return filesFeatures

    def getListOfWavFiles(self,location):
        import os.path
        import glob
        import fnmatch
        # determine the files to process
        files = []
        files.append(location)
        for f in files:
            # check what we have (file/path)
            if os.path.isdir(f):
                # use all files in the given path
                files = glob.glob(f + '/*.wav')
            else:
                # file was given, append to list
                files.append(f)
        # only process .wav files
        files = fnmatch.filter(files, '*.wav')
        files.sort()

        return files

    def __init__(self, frameSize = 4096, hopSize = frameSize/2):
        self.loader = essentia.standard.MonoLoader()
        self.writer = essentia.standard.MonoWriter()
        self.onsetRate = essentia.standard.OnsetRate()
        self.slicer = essentia.standard.Slicer()
        self.duration = essentia.standard.Duration()
        self.mfcc = essentia.standard.MFCC()
        self.spectrum = essentia.standard.Spectrum()
        self.w = essentia.standard.Windowing(type = 'hann')
        self.pitch = essentia.standard.PitchYinFFT()
        self.fft = essentia.standard.FFT()
        self.magnitude = essentia.standard.Magnitude()

        self.frameSize = frameSize
        self.hopSize = hopSize