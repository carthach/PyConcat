from __future__ import print_function
import essentia
import essentia.standard
import numpy as np
import matplotlib.pyplot as plt
import peakutils

enableDebug = False

class Extractor:
    frameSize = 2048
    hopSize = frameSize/2
    sampleRate = 44100.0

    i = 0

    def __init__(self, frameSize=2048, hopSize=frameSize/2, sampleRate=44100.0):
        self.frameSize = frameSize
        self.hopSize = hopSize
        self.sampleRate = 44100.0

        self.debugFile = open("processed_files.csv", 'w')

    def slice(self, onsetTimes, audio):
        """This can typically be faster than Essentia's slicer
        
        :param onsetTimes: Vector of onset times for slicing
        
        :param audio: Audio signal vector for slicing.
        
        :return: Audio segments.
        """

        segments = []

        for onsetTimeCounter in range(len(onsetTimes)):
            startOnsetTimeInSamples = int(onsetTimes[onsetTimeCounter] * self.sampleRate)

            endOnsetTimeInSamples = 0

            if onsetTimeCounter + 1 == len(onsetTimes):
                endOnsetTimeInSamples = len(audio)
            else:
                endOnsetTimeInSamples = int(onsetTimes[onsetTimeCounter + 1] * self.sampleRate)

            segment = audio[startOnsetTimeInSamples:endOnsetTimeInSamples]

            segments.append(segment)

        return segments

    def concatOnsets(self, sequence, corpusUnits, targetUnits, shouldStretch=True, shouldWindow=False):
        """Concatenate audio units back to back with optional time stretching to match the target
        Can also optionally window the audio
                
        :param sequence: list of indices into the corpusUnits
         
        :param corpusUnits: list of corpus unit audio signals
         
        :param targetUnits: list of target unit audio signals
         
        :param shouldStretch: stretch the corpus unit to match the target unit
        
        :param shouldWindow: apply a window to the signal
         
        :return: an audio signal
        """
        import pyrubberband as pyrb

        audio = []

        for i, item in enumerate(sequence):
            corpusUnit = corpusUnits[item]

            #Use Rubber Band to stretch the audio to match the target
            if shouldStretch:
                factor = len(corpusUnit) / float(len(targetUnits[i]))
                corpusUnit = pyrb.time_stretch(corpusUnit, 44100, factor)

            # Envelope the output audio using a hamming window
            if shouldWindow:
                window = np.hamming(len(audio))
                audio *= window


            audio = np.append(audio, corpusUnit)

        return audio


    def reSynth(self, sequence, ffts):
        """Resynthesise (using an IFFT) from the ffts, using a sequence of indices
        
        :param sequence: list of indices into the array of ffts
        
        :param ffts: list of ffts for each corpus unit
        
        :return: an audio signal 
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

    def writeAudio(self, audio, filename):
        """Write signal to the file system
        
        :param audio: audio signal
         
        :param fileName: output filename
        
        :param shouldWindow: 
        
        :return: None 
        """

        self.writer = essentia.standard.MonoWriter()
        self.writer.configure(filename=filename)

        audio = essentia.array(audio)

        self.writer(audio)

    def loadAudio(self, filename):
        """Load audio from a filename and return the audio vector
        
        :param filename: input filename
        
        :return: audio signal
        """

        audio = None

        if filename:
            loader = essentia.standard.MonoLoader(filename=filename)

            # and then we actually perform the loading:
            audio = loader()

        return audio

    def getListOfWavFiles(self, path):
        """Get list of wav files (or mp3s) in a folder
        
        :param path: directory containing soundfiles
        
        :return: a list of file names
        
        """
        import os.path
        import glob
        import fnmatch
        # determine the files to process
        files = []
        files.append(path)
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
        """Use a beattracker to return beat locations
        
        :param fileName: the file to load and extract beats from
         
        :return:
            ticks: the times in the file
        
            slices: the segmented audio units
        
            fileName: pass out the filename again
        """

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
        """Use an onset detector to return beat locations

        :param fileName: the file to load and extract beats from

        :return: 
            onsetTimes: the onset times in the file

            slices: the segmented audio units

            fileName: pass out the filename again

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
        """Write all the onsets to fileName_<no>.wav
        
        :param onsets: a list of audio signals
        
        :param fileName: the base filename
        
        :return: the list of filenames
        """
        import os

        fileNameOnly, fileNameExt = os.path.splitext(fileName)

        fileNames = []

        for onset in onsets:
            onsetFilename = fileNameOnly + "_" + str(self.outputFileCounter) + fileNameExt

            self.writeAudio(onset, onsetFilename)

            fileNames.append(onsetFilename)

            self.outputFileCounter = self.outputFileCounter+1

        return fileNames

    def extractFeatures(self, audio, scale="onsets", listOfFeatures=['Loudness', 'Centroid', 'Flatness', 'MFCC']):
        """Extract features from an audio vector.
        
        This tends to be pretty slow for onset based segmentation and retrieval
        
        :param audio: the audio to extract features from
        
        :param scale: the temporal scale we wish to use
        
        :return: 
        
            features: the list of audio features
        
            units: If FFT scale, then the fft frames also  
        """

        pool = essentia.Pool()
        medianPool = essentia.Pool()

        mfcc = centroid = flatness = energy = pitchYinFFT = spectralPeaks = hpcp = None

        if 'Centroid' in listOfFeatures:
            centroid = essentia.standard.Centroid(range=self.sampleRate / 2)
        if 'Flatness' in listOfFeatures:
            flatness = essentia.standard.Flatness()
        if 'Loudness' in listOfFeatures:
            loudness = essentia.standard.Loudness()
        if 'pitch' in listOfFeatures:
            loudness = essentia.standard.Loudness()
            pitchYinFFT = essentia.standard.PitchYinFFT()


        if 'MFCC' in listOfFeatures:
            mfcc = essentia.standard.MFCC(inputSize = self.frameSize/2+1)
        if 'HPCP' in listOfFeatures:
            spectralPeaks = essentia.standard.SpectralPeaks(orderBy="magnitude",
                                                            magnitudeThreshold=1e-05,
                                                            minFrequency=40,
                                                            maxFrequency=5000,
                                                            maxPeaks=10000)
            hpcp = essentia.standard.HPCP()

        fft = essentia.standard.FFT()
        magnitude = essentia.standard.Magnitude()
        w = essentia.standard.Windowing(type='blackmanharris62')

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

            if centroid is not None:
                centroidScalar = centroid(mag)
                pool.add("centroid", centroidScalar)
            if flatness is not None:
                flatnessScalar = flatness(mag)
                pool.add("flatness", flatnessScalar)
            if loudness is not None:
                loudnessScalar = loudness(frame)
                pool.add("loudness", loudnessScalar)
            if pitchYinFFT is not None:
                pitchScalar, pitchConfidenceScalar = pitchYinFFT(mag)
                # pool.add("pitch", pitchScalar)
                medianPool.add("pitch", pitchScalar)

            if mfcc is not None:
                mfcc_bands, mfccVector = mfcc(mag)
                pool.add("mfccs", mfccVector[1:])
            if hpcp is not None:
                frequencies, magnitudes = spectralPeaks(mag)
                hpcpVector = hpcp(frequencies, magnitudes)

                pool.add("pcps", hpcpVector)

                f.append(hpcpVector)

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
            # aggrPool = essentia.standard.PoolAggregator(defaultStats=['mean', 'var'])(pool)
            aggrPool = essentia.standard.PoolAggregator(defaultStats=['mean'])(pool)
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

    def analyseFile(self,file, writeOnsets, scale = "onsets"):
        """Extract onsets from a single file then extract features from all those onsets
        
        :param file: the file to analyse
        
        :param writeOnsets: whether you want to write the audio onsets to the filesystem
        
        :param scale: the temporal scale: None, spectral, onsets, beats
         
        :return:
        
            features : lists of lists of features
            
            units : list of audio signals corresponding to units
            
            unitTimes: the list of transient times from the audio signals
            
        """

        onsetTimes = []
        onsets = []
        fileName = file

        pool = essentia.Pool()

        print("Processing file: " + file)

        if enableDebug:
            self.debugFile.write(file + "\n")

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

    def analyseFiles(self,listOfFiles, writeOnsets=False, scale = "onsets"):
        """Perform segmentation and feature analysis on a list of files
        
        :param listOfFiles: list of filenames 
         
        :param writeOnsets: whether you want to write the audio onsets to the filesystem
        
        :param scale: the temporal scale: None, spectral, onsets, beats
        
        :return:
        
            features : lists of lists of features
            
            units : list of audio signals corresponding to units
            
            unitTimes: the list of transient times from the audio signals
            
        """
        features = []
        units = []
        unitTimes = []

        self.outputFileCounter = 0

        for file in listOfFiles:
            fileFeatures, fileUnits, fileUnitTimes = self.analyseFile(file, writeOnsets, scale)

            # features.append(fileFeatures)
            # ffts.append(fileFFTs)

            features += fileFeatures
            units += fileUnits
            unitTimes = np.append(unitTimes, fileUnitTimes)

        return features, units, unitTimes

