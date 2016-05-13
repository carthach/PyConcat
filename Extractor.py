from __future__ import print_function
import essentia
import essentia.standard
import numpy as np
import matplotlib.pyplot as plt
import peakutils

class Extractor:
    frameSize = 4096
    hopSize = frameSize/2

    def mySlicer(self, onsetTimes, audio):

        segments = []

        for onsetTimeCounter in range(len(onsetTimes)):
            startOnsetTimeInSamples = int(onsetTimes[onsetTimeCounter] * 44100.0)

            endOnsetTimeInSamples = 0

            if onsetTimeCounter + 1 == len(onsetTimes):
                endOnsetTimeInSamples = len(audio)
            else:
                endOnsetTimeInSamples = int(onsetTimes[onsetTimeCounter + 1] * 44100.0)

            segment = audio[startOnsetTimeInSamples:endOnsetTimeInSamples]

            segments.append(segment)

        return segments

    def synthResynth(self, audio):
        fft = essentia.standard.FFT()
        ifft = essentia.standard.IFFT()
        w = essentia.standard.Windowing(type = 'hann')

        overlapAdd = essentia.standard.OverlapAdd(frameSize = self.frameSize, hopSize = self.hopSize, gain=1.0/self.frameSize)
        audio_out = []

        i = 0
        for fstart in range(0, len(audio) - self.frameSize, self.hopSize):
            frame = audio[fstart:fstart + self.frameSize]
            fft_frame = fft(w(frame))

            ifft_frame = ifft(fft_frame)
            ifft_frame = overlapAdd(ifft_frame)

            audio_out = np.append(audio_out, ifft_frame)

            i = i + 1

        print("Number of frames: " + str(i))

        return audio_out




        return audio

    def resynthesise_audio(self, sequence, ffts):
        audio = []
        ifft = essentia.standard.IFFT()
        overlapAdd = essentia.standard.OverlapAdd(frameSize = self.frameSize, hopSize = self.hopSize, gain=1.0/self.frameSize)

        i = 0

        for unit in sequence:
            # fileIndex, onsetIndex, frameIndex = unit

            # clip = ifft(features[fileIndex][onsetIndex]["onsetFeatures"][frameIndex]["fft"])
            # clip = ifft(ffts[fileIndex][onsetIndex][frameIndex])
            clip = ifft(ffts[unit])
            # clip = self.w(clip)
            clip = overlapAdd(clip)

            audio = np.append(audio, clip)

            i = i + 1

        print("Number of units: " + str(i))

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
        mfcc = essentia.standard.MFCC(inputSize = self.frameSize/2+1)
        fft = essentia.standard.FFT()
        magnitude = essentia.standard.Magnitude()
        w = essentia.standard.Windowing(type='hann')
        yin = essentia.standard.PitchYinFFT()
        energy = essentia.standard.Energy()

        features = []
        ffts = []

        for fstart in range(0, len(audio) - self.frameSize, self.hopSize):
            frame = audio[fstart:fstart + self.frameSize]

            fft_frame = fft(w(frame))
            mag = magnitude(fft_frame)

            pitch, pitchConfidence = yin(mag)

            mfcc_bands, mfcc_coeffs = mfcc(mag)
            e = energy(frame)

            f = pitch + e + mfcc_coeffs

            features.append(f)
            ffts.append(fft_frame)

        # for frame in essentia.standard.FrameGenerator(audio, frameSize=1024, hopSize=512):
        #     mfcc_bands, mfcc_coeffs = self.mfcc(self.spectrum(self.w(frame)))
        #     pool.add('lowlevel.mfcc', mfcc_coeffs)
        #     pool.add('lowlevel.mfcc_bands', mfcc_bands)

        return features, ffts

    def loadAudio(self, fileName):

        audio = None

        if fileName:
            loader = essentia.standard.MonoLoader(filename=fileName)

            # and then we actually perform the loading:
            audio = loader()

        return audio


    def extractOnsets(self,fileName):
        """
        Extract and return a vector of onsets as audio vectors
        :param file:
        :return:
        """

        slices = None
        onsetTimes = None

        onsetRate = essentia.standard.OnsetRate()
        duration = essentia.standard.Duration()

        if fileName:
            audio = self.loadAudio(fileName)
            onsetRateResult = onsetRate(audio)

            onsetTimes, onsetRate  = onsetRateResult
            endTimes = onsetTimes[1:]
            d = duration(audio)
            endTimes = np.append(endTimes, d)
            endTimes = essentia.array(endTimes)

            slicer = essentia.standard.Slicer(startTimes=onsetTimes, endTimes=endTimes)

            slices = slicer(audio)

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

        features = []
        ffts = []

        for onsetTime, onset in zip(onsetTimes, onsets):
            onsetFeatures, onsetFFTs = self.extractFeatures(onset)


            # features.append(onsetFeatures)
            # ffts.append(onsetFFTs)

            features = features + onsetFeatures
            ffts = ffts + onsetFFTs

        return features, ffts

    def analyseFile(self,file, writeOnsets, onsetBased = True):
        """
        Extract onsets from a single file then extract features from all those onsets
        :param file:
        :return:
        """

        onsetTimes = []
        onsets = []
        fileName = file

        if onsetBased:
            onsetTimes, onsets, fileName = self.extractOnsets(file)
        else:
            onsetTimes.append(0.0)
            audio = self.loadAudio(file)
            onsets.append(audio)

        print("Processing file: " + file)

        if (writeOnsets):
            fileNames = self.writeOnsets(onsets, file)

        features = []
        ffts = []

        for onsetTime, onset in zip(onsetTimes, onsets):
            onsetFeatures, onsetFFTs = self.extractFeatures(onset)

            # features.append(onsetFeatures)
            # ffts.append(onsetFFTs)

            features = features + onsetFeatures
            ffts = ffts + onsetFFTs

        return features, ffts

    def analyseFiles(self,listOfFiles):
        """
        Perform onset detection and extract features from all the onsets from all the files
        :param listOfFiles:
        :return:
        """
        features = []
        ffts = []

        for file in listOfFiles:
            fileFeatures, fileFFTs = self.analyseFile(file, False, False)

            # features.append(fileFeatures)
            # ffts.append(fileFFTs)

            features = features + fileFeatures
            ffts = ffts + fileFFTs

        return features, ffts

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
                files = files + glob.glob(f + '/*.mp3')
            else:
                # file was given, append to list
                files.append(f)
        # only process .wav files
        # files = fnmatch.filter(files, '*.wav')
        files.sort()

        return files
