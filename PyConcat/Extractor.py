from __future__ import print_function
import essentia
import essentia.standard
import numpy as np
import matplotlib.pyplot as plt
import peakutils
import glob
import fnmatch
import os
import madmom
import librosa
from madmom.audio.filters import LogarithmicFilterbank
from multiprocessing import Process, Queue
from multiprocessing import Pool

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

    def concatOnsets(self, sequence, corpusUnits, targetUnits, stretchUnits=False, windowUnits=False):
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
            if stretchUnits:
                factor = len(corpusUnit) / float(len(targetUnits[i]))
                corpusUnit = pyrb.time_stretch(corpusUnit, 44100, factor)

            # Envelope the output audio using a hamming window
            if windowUnits:
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
            # loader = essentia.standard.MonoLoader(filename=filename)
            #
            # # and then we actually perform the loading:
            # audio = loader()

            #Essentia's loader (above)  has a bug that doesn't close files
            #It causes problems processing large number of files, use madmom instead
            # audio, sample_rate = madmom.audio.signal.load_wave_file(filename, num_channels=1)
        
            y, sr = librosa.load(filename, sr=None)

            audio = essentia.array(y)

        return audio

    def getListOfFiles(self, path, filterPattern=""):
        """Get list of wav files (or mp3s) in a folder

        :param path: directory containing soundfiles

        :return: a list of file names

        """
        files = []

        #For the recursive functionality we need a list of paths
        if type(path) is not list:
            path = [path]

        for f in path:
            # check what we have (file/path)
            if os.path.isdir(f):
                # use all files in the given path
                files = glob.glob(f + '/' + filterPattern)
            else:
                # file was given, append to list
                files.append(f)

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

    def extractOnsetsMadmom(self, filename, method="CNN"):
        """Use Madmom's Superflux or CNN detector to return onset locations

        :param fileName: the file to load and extract beats from

        :return:
            onsetTimes: the onset times in the file

            slices: the segmented audio units

            fileName: pass out the filename again

        """
        onsetFunctionExtractor = None
        peakPicker = None

        #Choose an onset detection function
        if "CNN" in method:
            onsetFunctionExtractor = madmom.features.onsets.CNNOnsetProcessor(fps=100)
            peakPicker = madmom.features.onsets.OnsetPeakPickingProcessor(threshold=0.54, smooth=0.05)
        else:
            # onsetFunctionExtractor = madmom.features.onsets.SpectralOnsetProcessor(onset_method='superflux',
            #                                                     filterbank = LogarithmicFilterbank,
            #                                                     num_bands = 24, log = np.log10,
            #                                                     fmin=30, fmax=17000,
            #                                                     mul=1, add=1,
            #                                                     diff_ratio=0.5,
            #                                                     diff_max_bins=3,
            #                                                     positive_diffs=True
            #                                                     )
            # peakPicker = madmom.features.onsets.OnsetPeakPickingProcessor(threshold=1.1, pre_max=0.01,
            #                                                               post_max=0.05, pre_avg=0.15,
            #                                                               post_avg=0, combine=0.03, delay=0)

            onsetFunctionExtractor = madmom.features.onsets.SpectralOnsetProcessor(onset_method='superflux',
                                                                                   filterbank=LogarithmicFilterbank,
                                                                                   num_bands=24, log=np.log10,
                                                                                   fmin=27.5, fmax=16000,
                                                                                   mul=1, add=1,
                                                                                   diff_ratio=0.5,
                                                                                   diff_max_bins=3,
                                                                                   positive_diffs=True
                                                                                   )
            #Old SupeFlux Params
            peakPicker = madmom.features.onsets.OnsetPeakPickingProcessor(threshold=1.25, pre_max=0.03,
                                                                          post_max=0.03, pre_avg=0.10,
                                                                          post_avg=0.07, combine=0.03, delay=0)

        #get the onsets
        onsetFunction = onsetFunctionExtractor(filename)
        onsetTimes = peakPicker(onsetFunction)

        return onsetTimes

    def extractOnsetsEssentia(self,filename):
        """Use Essentia's Onset Rate onset detector to return beat locations

        :param fileName: the file to load and extract beats from

        :return:
            onsetTimes: the onset times in the file

            slices: the segmented audio units

            fileName: pass out the filename again

        """
        slices = None
        onsetTimes = None

        onsetRate = essentia.standard.OnsetRate()

        audio = self.loadAudio(filename)
        onsetTimes, onsetRate = onsetRate(audio)

        return onsetTimes

    def extractAndSliceOnsets(self, filename, method="Essentia"):
        """Use an onset detector to return beat locations

        :param fileName: the file to load and extract beats from

        :return:
            onsetTimes: the onset times in the file

            slices: the segmented audio units

            fileName: pass out the filename again

        """
        if filename:
            slices = None

            audio = self.loadAudio(filename)

            onsetTimes = None

            if "Madmom" in method:
                onsetTimes = self.extractOnsetsMadmom(filename, method=method)
            else:
                onsetTimes = self.extractOnsetsEssentia(filename)


            endTimes = onsetTimes[1:]

            duration = essentia.standard.Duration()
            d = duration(audio)

            endTimes = np.append(endTimes , d)
            endTimes = essentia.array(endTimes)

            slices = self.slice(onsetTimes, audio)

        return onsetTimes, slices, filename

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

    def extractFeatures(self, audio, scale="onsets", listOfFeatures=['Loudness', 'Centroid', 'Flatness', 'BFCC']):
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

        centroid = flatness = loudness = pitchYinFFT = None
        bfcc = mfcc = spectralPeaks = hpcp = None

        if 'Centroid' in listOfFeatures:
            centroid = essentia.standard.Centroid(range=self.sampleRate / 2)
        if 'Flatness' in listOfFeatures:
            flatness = essentia.standard.Flatness()
        if 'Loudness' in listOfFeatures:
            loudness = essentia.standard.Loudness()
        if 'Pitch' in listOfFeatures:
            pitchYinFFT = essentia.standard.PitchYinFFT()


        if 'BFCC' in listOfFeatures:
            bfcc = essentia.standard.BFCC(inputSize = self.frameSize/2+1)
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
                pool.add("Centroid", centroidScalar)
            if flatness is not None:
                flatnessScalar = flatness(mag)
                pool.add("Flatness", flatnessScalar)
            if loudness is not None:
                loudnessScalar = loudness(frame)
                pool.add("Loudness", loudnessScalar)
            if pitchYinFFT is not None:
                pitchScalar, pitchConfidenceScalar = pitchYinFFT(mag)
                # pool.add("pitch", pitchScalar)
                medianPool.add("Pitch", pitchScalar)

            import time

            startTime = time.time()

            if bfcc is not None:
                bfcc_bands, bfccVector = bfcc(mag)
                pool.add("BFCC", bfccVector[1:])
            if mfcc is not None:
                mfcc_bands, mfccVector = mfcc(mag)
                pool.add("MFCC", mfccVector[1:])
            if hpcp is not None:
                frequencies, magnitudes = spectralPeaks(mag)
                hpcpVector = hpcp(frequencies, magnitudes)

                pool.add("HPCP", hpcpVector)

                f.append(hpcpVector)

            elapsedTime = time.time() - startTime

            x = pool.descriptorNames()

            #If we are spectral based we need to return the fft frames as units and the framewise features
            if scale is "spectral":
                units.append(fft_frame)

                frameFeatures = []

                """
                We do it this roundabout way to retain the order that user wants in listOfFeatures
                """
                for feature in listOfFeatures:
                    for descriptor in pool.descriptorNames():
                        if feature in descriptor:
                            frameFeatures = np.append(frameFeatures, (pool[descriptor]))

                    for descriptor in medianPool.descriptorNames():
                        if feature in descriptor:
                            frameFeatures = np.append(frameFeatures, (medianPool[descriptor]))

                features.append(frameFeatures)
                pool.clear()
                medianPool.clear()


        #Now we get all the stuff out of the pool
        if scale is not "spectral":
            # aggrPool = essentia.standard.PoolAggregator(defaultStats=['mean', 'var'])(pool)
            aggrPool = essentia.standard.PoolAggregator(defaultStats=['mean'])(pool)
            medianAggrPool = essentia.standard.PoolAggregator(defaultStats=['median'])(medianPool)

            """
            We do it this roundabout way to retain the order that user wants in listOfFeatures
            """
            for feature in listOfFeatures:
                for aggrFeature in aggrPool.descriptorNames():
                    if feature in aggrFeature:
                        if "mean" or "variance" in feature:
                            features = np.append(features, aggrPool[aggrFeature])
                        else:
                            features += aggrPool[aggrFeature][0]

                #Median based features (i.e. pitch)
                for medianFeature in medianAggrPool.descriptorNames():
                    if feature in medianFeature:
                        if "median" in medianFeature:
                            features = np.append(features, medianAggrPool[medianFeature])
                        else:
                            features += medianAggrPool[medianFeature][0]

            aggrPool.merge(medianAggrPool)

        #Return features, and if it's spectral return the frames as units
        return features, units, pool

    def analyseFile(self,file, writeOnsets, scale = "onsets", yamlOutputFile="", onsetDetection="", listOfFeatures=['Loudness', 'Centroid', 'Flatness', 'BFCC']):
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

        filePool = essentia.Pool()

        print("Processing file: " + file)

        if enableDebug:
            self.debugFile.write(file + "\n")

        #Extract onsets or add the audio as a single onset
        print("    Onset Detection and Segmentation...")
        if scale == "beats":
            onsetTimes, onsets, fileName = self.extractBeats(file)
        elif scale == "onsets":
            onsetTimes, onsets, fileName = self.extractAndSliceOnsets(file, method=onsetDetection)
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
            onsetFeatures, onsetFFTs, onsetPool = self.extractFeatures(onset, scale, listOfFeatures=listOfFeatures)

            #If it's not onset based then spectra are the units, append
            if scale is "spectral":
                units += onsetFFTs
                features += onsetFeatures
            else:
                features.append(onsetFeatures)

            onsetPool.add("onsetTimes", onsetTime)
            filePool.merge(onsetPool, "append")

        if scale is not "spectral":
            units = onsets

        if yamlOutputFile != "":
            essentia.standard.YamlOutput(filename=yamlOutputFile)(filePool)

        return features, units, onsetTimes

    def analyseFiles(self,listOfFiles, writeOnsets=False, scale = "onsets", yamlOutputFolder="", onsetDetection="", listOfFeatures=['Loudness', 'Centroid', 'Flatness', 'BFCC']):
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

            yamlOutputFile = ""
            if yamlOutputFolder != "":
                import os
                baseFilename = os.path.splitext(os.path.basename(file))[0]
                yamlOutputFile = yamlOutputFolder + "/" + baseFilename + ".yaml"

            fileFeatures, fileUnits, fileUnitTimes = self.analyseFile(file, writeOnsets, scale, yamlOutputFile=yamlOutputFile, onsetDetection=onsetDetection, listOfFeatures=listOfFeatures)

            # features.append(fileFeatures)
            # ffts.append(fileFFTs)

            features += fileFeatures
            units += fileUnits
            unitTimes = np.append(unitTimes, fileUnitTimes)

        return features, units, unitTimes