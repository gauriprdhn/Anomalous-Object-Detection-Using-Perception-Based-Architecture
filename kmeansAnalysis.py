from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class kMeansAnalysis:
    
    def __init__(self,data,labels,nClusters=1):
        self.X = data
        self.n = nClusters
        self.y = labels

    def gettingKMeansPredictions(self):
        kmeans = KMeans(init = "k-means++", n_clusters=self.n, n_init = 50)
        kmeans.fit(self.X)
        y_kmeans = kmeans.predict(self.X)
        return kmeans , y_kmeans
    
    def clusterVisualization(self,y_kmeans):
        visited = set()
        for i in range(len(self.X)):
            if y_kmeans[i] == self.y[i] and y_kmeans[i] not in visited:
                plt.imshow(self.X[i])
                title = "For Category = " + str(y_kmeans[i]) 
                plt.title(title)
                plt.show()
                visited.add(y_kmeans[i])
                
    def plotClusters3D(self,model,y_kmeans):
        _ = plt.figure(figsize = (16, 12))
        ax = plt.axes(projection ="3d")
        ax.grid(b = True, color ='grey', 
            linestyle ='-.', linewidth = 0.3, 
            alpha = 0.2) 
        # Creating plot
        LABEL_COLOR_MAP = {0 : 'r', 1 : 'k', 2: 'm', 3: 'c', 4:'b', 5:'g'}      
        label_color = [LABEL_COLOR_MAP[l] for l in y_kmeans]
        ax.scatter3D(self.X[:,0], self.X[:,1], self.X[:,2], alpha = 0.8, c = label_color, s=50)
        centers = model.cluster_centers_
        ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2],c='black', s=200, alpha=0.5)
        plt.show()

    def plotClusters2D(self,model,y_kmeans):
        _, ax = plt.subplots(1,1,figsize=(16,12))
        LABEL_COLOR_MAP = {0 : 'r', 1 : 'k', 2: 'm', 3: 'c', 4:'b', 5:'g'}      
        label_color = [LABEL_COLOR_MAP[l] for l in y_kmeans]
        ax.scatter(self.X[:, 0], self.X[:, 1], alpha = 0.8, c = label_color, s=50)
        centers = model.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.show()