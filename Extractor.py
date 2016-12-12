from __future__ import print_function
import essentia
import essentia.standard
import numpy as np
import matplotlib.pyplot as plt
import peakutils

class Extractor:
    frameSize = 2048
    hopSize = frameSize/2
    sampleRate = 44100

    def slice(self, onsetTimes, audio):
        """
        This can typically be faster than Essentia's slicer
        :param onsetTimes: Vector of onset times for slicing
        :param audio: Audio signal vector for slicing.
        :return: Audio segments.
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

    def pad(self, audio, padLength):
        audio

    def concatOnsets(self, sequence, corpusUnits, targetUnits):
        """

        :param sequence:
        :param units:
        :param unitTimes:
        :return:
        """
        import pyrubberband as pyrb
        shouldStretch = True

        audio = []

        for i, item in enumerate(sequence):
            corpusUnit = corpusUnits[item]

            if shouldStretch:
                factor = len(corpusUnit) / float(len(targetUnits[i]))

                corpusUnit = pyrb.time_stretch(corpusUnit, 44100, factor)

            audio = np.append(audio, corpusUnit)

        return audio


    def reSynth(self, sequence, ffts):
        """
        Resynthesise from the ffts, using a sequence of indices
        :param sequence:
        :param ffts:
        :return:
        """

        audio = []

        ifft = essentia.standard.IFFT(size = self.frameSize)

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

    def extractBeats(self, fileName):

        slices = None
        ticks = None

        beatTracker = essentia.standard.BeatTrackerDegara()
        duration = essentia.standard.Duration()

        if fileName:
            audio = self.loadAudio(fileName)

            ticks = beatTracker(audio)

            endTimes = ticks[1:]
            d = duration(audio)
            endTimes = np.append(endTimes, d)
            endTimes = essentia.array(endTimes)

            # slicer = essentia.standard.Slicer(startTimes=onsetTimes, endTimes=endTimes)
            # slices = slicer(audio)

            slices = self.slice(ticks, audio)

        return ticks, slices, fileName

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
            endTimes = np.append(endTimes , d)
            endTimes = essentia.array(endTimes)

            # slicer = essentia.standard.Slicer(startTimes=onsetTimes, endTimes=endTimes)
            # slices = slicer(audio)

            slices = self.slice(onsetTimes, audio)

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

    def extractFeatures(self,audio, scale = "beats"):
        """
        Extract features from an audio vector.
        This tends to be pretty slow for onset based segmentation and retrieval
        :param audio:
        :return:
        """

        pool = essentia.Pool()
        medianPool = essentia.Pool()
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

        centroid = essentia.standard.Centroid(range = self.sampleRate/2)

        hpcp = essentia.standard.HPCP()

        features = []
        units = []

        f = []

        # #Manual framecutting is faster than Essentia in Python
        # for fstart in range(0, len(audio) - self.frameSize, self.hopSize):
        #     #Get the frame
        #     frame = audio[fstart:fstart + self.frameSize]

        for frame in essentia.standard.FrameGenerator(audio, frameSize=self.frameSize, hopSize=self.hopSize):

            #FFT and Magnitude Spectrum
            fft_frame = fft(w(frame))
            mag = magnitude(fft_frame)

            #Pitch
            pitch, pitchConfidence = yin(mag)

            #MFCCs
            mfcc_bands, mfcc_coeffs = mfcc(mag)

            #Energy
            e = energy(frame)

            c = centroid(mag)

            #Key
            frequencies, magnitudes = spectralPeaks(mag)
            pcps = hpcp(frequencies, magnitudes)
            f.append(pcps)

            pool.add("energy", e)
            pool.add("centroid", c)
            pool.add("pcps", pcps)
            # pool.add("mfccs", mfcc_coeffs[1:])
            pool.add("mfccs", mfcc_coeffs)

            medianPool.add("pitch", pitch)

            #If we are spectral based we need to return the fft frames as units and the framewise features
            if scale is "spectral":
                units.append(fft_frame)

                frameFeatures = []
                for descriptor in pool.descriptorNames():
                    frameFeatures = np.append(frameFeatures, (pool[descriptor]))

                for descriptor in medianPool.descriptorNames():
                    frameFeatures = np.append(frameFeatures, (medianPool[descriptor]))

                features.append(frameFeatures)
                pool.clear()
                medianPool.clear()


        #Now we get all the stuff out of the pool
        if scale is not "spectral":
            aggrPool = essentia.standard.PoolAggregator(defaultStats=['mean', 'var'])(pool)
            medianAggrPool = essentia.standard.PoolAggregator(defaultStats=['median'])(medianPool)

            for feature in aggrPool.descriptorNames():
                if "mean" or "variance" in feature:
                    features = np.append(features, aggrPool[feature])
                else:
                    features += aggrPool[feature][0]

            for feature in medianAggrPool.descriptorNames():
                if "median" in feature:
                    features = np.append(features, medianAggrPool[feature])
                else:
                    features += medianAggrPool[feature][0]

        #Return features, and if it's spectral return the frames as units
        return features, units

    def analyseFile(self,file, writeOnsets, scale = "beats"):
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
        print("    Onset Detection and Segmentation...")
        if scale is "beats":
            onsetTimes, onsets, fileName = self.extractBeats(file)
        elif scale is "onsets":
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
            onsetFeatures, onsetFFTs = self.extractFeatures(onset, scale)

            #If it's not onset based then spectra are the units, append
            if scale is "spectral":
                units += onsetFFTs
                features += onsetFeatures
            else:
                features.append(onsetFeatures)

        if scale is not "spectral":
            units = onsets

        return features, units, onsetTimes

    def analyseFiles(self,listOfFiles, writeOnsets=False, scale = "beats"):
        """
        Perform onset detection and extract features from all the onsets from all the files
        :param listOfFiles:
        :return:
        """
        features = []
        units = []
        unitTimes = []

        for file in listOfFiles:
            fileFeatures, fileUnits, fileUnitTimes = self.analyseFile(file, writeOnsets, scale)

            # features.append(fileFeatures)
            # ffts.append(fileFFTs)

            features += fileFeatures
            units += fileUnits
            unitTimes = np.append(unitTimes, fileUnitTimes)

        return features, units, unitTimes

