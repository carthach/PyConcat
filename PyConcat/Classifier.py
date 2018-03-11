from sklearn.neighbors import KNeighborsClassifier

class Classifier:

    def __init__(self):
        pass

    def trainClassifier(self):
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X, y)

    def classifyInstance(self):
        pass


