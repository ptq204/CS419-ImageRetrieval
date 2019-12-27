from sklearn.cluster import MiniBatchKMeans

class Cluster:
    def __init__(self, num_clusters):
        self.numClusters = num_clusters

    def cluster(self, X):
        kmeans = MiniBatchKMeans(n_clusters = self.numClusters, random_state = 0).fit(X)
        return kmeans
