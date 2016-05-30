from __future__ import print_function
import essentia
import essentia.standard
import numpy as np
import matplotlib.pyplot as plt
import peakutils

class Extractor:
    x = 0
    frameSize = 4096
    hopSize = frameSize/2

    def mySlicer(self, onsetTimes, audio):
        """
        This can typically be faster than Essentia's slicer
        :param onsetTimes:
        :param audio:
        :return:
        """

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
        """
        Just to test that basic FFT synthesis / resynthesis works
        :param audio:
        :return:
        """
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

    def concat(self, sequence, units):
        audio = []

        for i in sequence:
            audio = np.append(audio, units[i])

        return audio



    def reSynth(self, sequence, ffts):
        """
        Resynthesise from the ffts, using a sequence of indices
        :param sequence:
        :param ffts:
        :return:
        """

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

    def writeAudio(self, audio, fileName):
        """
        Write a vector of audio to fileName
        :param audio:
        :param fileName:
        :return:
        """
        self.writer = essentia.standard.MonoWriter()
        self.writer.configure(filename=fileName)
        audio = essentia.array(audio)
        self.writer(audio)

    def loadAudio(self, fileName):
        """
        Load audio from a fileName and return the audio vector
        :param fileName:
        :return:
        """
        audio = None

        if fileName:
            loader = essentia.standard.MonoLoader(filename=fileName)

            # and then we actually perform the loading:
            audio = loader()

        return audio

    def getListOfWavFiles(self,location):
        """
        Get list of wav files (or mp3s) in a folder
        :param location:
        :return:
        """
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


    ######################

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

            # slicer = essentia.standard.Slicer(startTimes=onsetTimes, endTimes=endTimes)
            # slices = slicer(audio)

            slices = self.mySlicer(onsetTimes, audio)

        return onsetTimes, slices, fileName

    def writeOnsets(self, onsets, fileName):
        """
        Write all the onsets to fileName_<no>.wav
        :param onsets:
        :param fileName:
        :return:
        """
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

    def extractFeatures(self,audio, onsetBased = True):
        """
        Extract features from an audio vector.
        This tends to be pretty slow for onset based segmentation and retrieval
        :param audio:
        :return:
        """

        pool = essentia.Pool()
        mfcc = essentia.standard.MFCC(inputSize = self.frameSize/2+1)
        fft = essentia.standard.FFT()
        magnitude = essentia.standard.Magnitude()
        w = essentia.standard.Windowing(type='blackmanharris62')
        yin = essentia.standard.PitchYinFFT()
        energy = essentia.standard.Energy()

        spectralPeaks = essentia.standard.SpectralPeaks(orderBy =  "magnitude",
                                                        magnitudeThreshold = 1e-05,
                                                        minFrequency = 40,
                                                        maxFrequency = 5000,
                                                        maxPeaks = 10000)

        hpcp = essentia.standard.HPCP()

        features = []
        units = []

        f = []

        self.x += 1

        if self.x == 13:
            print("hello")

        # #Manual framecutting is faster than Essentia in Python
        # for fstart in range(0, len(audio) - self.frameSize, self.hopSize):
        #     #Get the frame
        #     frame = audio[fstart:fstart + self.frameSize]

        for frame in essentia.standard.FrameGenerator(audio, frameSize=self.frameSize, hopSize=self.hopSize):

            #FFT and Magnitude Spectrum
            fft_frame = fft(w(frame))
            mag = magnitude(fft_frame)

            #Pitch
            #pitch, pitchConfidence = yin(mag)

            #MFCCs
            mfcc_bands, mfcc_coeffs = mfcc(mag)

            #Energy
            e = energy(frame)

            #Key
            frequencies, magnitudes = spectralPeaks(mag)
            pcps = hpcp(frequencies, magnitudes)
            f.append(pcps)

            pool.add("PCPS", pcps)

            #If we are spectral based we need to return the fft frames as units
            if not onsetBased:
                units.append(fft_frame)

        if onsetBased:
            aggrPool = essentia.standard.PoolAggregator(defaultStats=['mean', 'var'])(pool)
            if "PCPS.mean" in aggrPool.descriptorNames():
                features = aggrPool["PCPS.mean"]
            elif "PCPS" in aggrPool.descriptorNames():
                features = aggrPool["PCPS"][0]
        else:
            features = pool["PCPS"]

        return features, units

    def analyseFile(self,file, writeOnsets, onsetBased = True):
        """
        Extract onsets from a single file then extract features from all those onsets
        :param file:
        :return:
        """

        onsetTimes = []
        onsets = []
        fileName = file

        pool = essentia.Pool()

        print("Processing file: " + file)

        #Extract onsets or add the audio as a single onset
        if onsetBased:
            print("    Onset Detection and Segmentation...")
            onsetTimes, onsets, fileName = self.extractOnsets(file)
        else:
            onsetTimes.append(0.0)
            audio = self.loadAudio(file)
            onsets.append(audio)

        #Optionally write these onsets out
        if (writeOnsets):
            fileNames = self.writeOnsets(onsets, file)

        features = []
        units = []

        print("    Feature Extraction...")

        for onsetTime, onset in zip(onsetTimes, onsets):
            onsetFeatures, onsetFFTs = self.extractFeatures(onset, onsetBased)

            features.append(onsetFeatures)

            #If it's not onset based then spectra are the units, append
            if not onsetBased:
                units = units + onsetFFTs

        if onsetBased:
            units = onsets

        return features, units

    def analyseFiles(self,listOfFiles, onsetBased = True):
        """
        Perform onset detection and extract features from all the onsets from all the files
        :param listOfFiles:
        :return:
        """
        features = []
        units = []

        for file in listOfFiles:
            fileFeatures, fileUnits = self.analyseFile(file, False, onsetBased)

            # features.append(fileFeatures)
            # ffts.append(fileFFTs)

            features += fileFeatures
            units += fileUnits

        return features, units

