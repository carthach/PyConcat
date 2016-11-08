import numpy as np
from scipy.spatial import distance

class HMM(object):

    def viterbi(self):
        ''' Given sequence of emissions, return the most probable path
            along with log2 of its probability.  Just like viterbi(...)
            but in log2 domain. '''
        nrow, ncol = len(self.stateMatrix), len(self.emissionMatrix)

        mat = np.zeros(shape=(nrow, ncol), dtype=float)  # prob
        matTb = np.zeros(shape=(nrow, ncol), dtype=int)  # backtrace

        concatenationWeight = 0.0
        targetWeight = 1.0

        # Fill in first column
        for i in xrange(0, nrow):
            mat[i, 0] = self.emissionMatrix[0, i]

        # Fill in rest of prob and Tb tables
        for j in xrange(1, ncol): #For every target unit
            for i in xrange(0, nrow): #for every corpus unit
                targetCost = self.emissionMatrix[j, i] #Get target cost
                concatenationCost = self.stateMatrix[i, 0]

                mx, mxi = mat[0, j - 1] + (concatenationWeight * concatenationCost) + (targetWeight * targetCost), 0

                for i2 in xrange(1, nrow):
                    concatenationCost = self.stateMatrix[i, i2]

                    #Make adjacent costs 0
                    # if np.abs(i2 - i) == 1:
                    #     concatenationCost = 0

                    pr = mat[i2, j-1] + (concatenationWeight * concatenationCost) + (targetWeight * targetCost)

                    if pr < mx:
                        mx, mxi = pr, i2

                mat[i, j], matTb[i, j] = mx, mxi

        # Find final state with maximal probability
        omx, omxi = mat[0, ncol-1], 0

        for i in xrange(1, nrow):
            if mat[i, ncol-1] < omx:
                omx, omxi = mat[i, ncol-1], i

        # Backtrace
        i, p = omxi, [omxi]
        for j in xrange(ncol-1, 0, -1):
            i = matTb[i, j]
            p.append(i)

        p = np.flipud(p)

        # return omx, p # Return probability and path
        return p #Just return path

    def __init__(self, targetFeatures, corpusFeatures):
        self.emissionMatrix = distance.cdist(targetFeatures, corpusFeatures, 'euclidean')

        corpusFeatures = corpusFeatures[:,0:2]

        self.stateMatrix = distance.cdist(corpusFeatures, corpusFeatures, 'euclidean')


        # self.stateMatrix[self.stateMatrix == 0] = 1.0


