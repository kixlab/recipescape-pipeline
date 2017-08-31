import numpy as np
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

#load files
ids = np.load("recipe_ids.dat")
dist_mat = np.load("dist_matrix.dat")
dist_mat = dist_mat + dist_mat.T - np.diag(dist_mat.diagonal())
#dist_mat.shape

#make 2d embedding
mds2 = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=4)
embed2d = mds2.fit(dist_mat).embedding_ #xy-coordinates converted from distance matrix

X = embed2d #lazy to type embed2d from here

#visualize data
plt.scatter(X[:,0], X[:,1])
plt.show()

# generate the linkage matrix
Z = linkage(X, 'ward')
c, coph_dists = cophenet(Z, pdist(X))

#draw dendogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

#visualize nicely
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=16,
)
plt.show()

#elbow method
# last = Z[-10:, 2]
# last_rev = last[::-1]
# idxs = np.arange(1, len(last) + 1)
# plt.plot(idxs, last_rev)

# acceleration = np.diff(last, 2)  # 2nd derivative of the distances
# acceleration_rev = acceleration[::-1]
# plt.plot(idxs[:-2] + 1, acceleration_rev)
# plt.show()
# k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
# print("clusters:", k)

#CLUSTER MEMBERSHIP 
max_d = 100 #this is a parameter determined from examining dendogram
clusters = fcluster(Z, max_d, criterion='distance')

#plot with cluster colors
plt.figure(figsize=(10, 8))
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
plt.show()


result = np.column_stack((ids.T, embed2d, clusters))
result.dump("clustering_result.dat"

import math
def dist(x1,y1,x2,y2):
    return math.hypot(x2-x1, y2-y1)

rep_points = []
for clust in list(set(clusters)):
    points = []
    for i in result:
        if int(i.item(0,3)) == clust:
            points.append(i)
    
    x = [float(p.item(0,1)) for p in points]
    y = [float(p.item(0,2)) for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))
    
    min_val = 99999
    min_p = points[0]
    
    for p in points:
        if dist(float(p.item(0,1)),float(p.item(0,2)), centroid[0], centroid[1]) < min_val:
            min_val = dist(float(p.item(0,1)),float(p.item(0,2)), centroid[0], centroid[1])
            min_p = p
    
    rep_points.append(min_p)

rep_points = np.array(rep_points)
rep_points.dump("rep_points.dat")
