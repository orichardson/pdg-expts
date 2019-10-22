import numpy as np
from numpy import linalg
import scipy
import pandas as pd
from domains import Domain, Link

# Generate Synthetic Data
from sklearn.datasets import make_moons, make_circles, make_classification, load_wine
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessClassifier


from tropical import perron, maxevec

pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(precision=3, suppress=True)



def round_at(number, k=1):
    return round(number * k) / k


#Make data set: synthetic 1
# X, y = pd.DataFrame(make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1,
#                             n_clusters_per_class=1, n_samples= 200)) # default: samples=100
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)
# def bin( x ):
#     x0, x1 = x
#     return (round_at(x0), round_at(x1))


#Make data set: wines
wine_ds = load_wine()
X = pd.DataFrame(wine_ds.data, columns=wine_ds.feature_names)
y = wine_ds.target
avgs = np.mean(X,axis=0)
def bin( x ):
    return tuple(np.sign(x - avgs))\
#from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# def bin( x ):
#     return tuple(np.sign(x - avgs)[:3])




# linearly_separable = (X, y)
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=42)
        
    

exp = Domain.from_util(dict(enumerate(y_train) )).comp
X_link = Link.from_dict( dict(enumerate( bin(x) for x in X_train.values )) ).Pr # Very dumb.

T = linalg.pinv(X_link)
T_labeled = pd.DataFrame(T, columns = X_link.index, index=X_link.columns)

X_comp = X_link @ exp @ X_link.T
X_comp2 = T_labeled.T @ exp @ T_labeled
# X_comp2

# U, Σ, V = linalg.svd(X_comp)
# linalg.matrix_rank(X_comp)

X_comp_SR = np.exp(X_comp2)

# calculate Peron-Frobenius Vector, and maxevec
#perron(X_comp_SR.to_numpy())
#maxevec(X_comp_SR.to_numpy())

_, zs = maxevec(X_comp_SR.to_numpy())

X_comp_SR
#plt.matshow(X_comp_SR)
############### PLOTTING ##############
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
plt.style.use('ggplot')

########## bar plots for each feature #####
#print(wine_ds.feature_names[:3])
x_pos = list(range(len(zs)))
plt.bar(x_pos, zs, color='green')
plt.show()
###############
#plot_indices = zip(*X_comp_SR.columns)
# 
# fig, ax = plt.subplots()
# cm = plt.cm.RdBu       
# cm_bright = ListedColormap(['#FF0000', '#0000FF'])
# 
# ax.tricontourf(x1s, x2s, zs, 100, cmap=cm, alpha=.8)
# ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
# #ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
# 
# ax.scatter(x1s,x2s, c=zs, s=80, cmap =cm, alpha=0.2)
# plt.show()
# 
# so, this is shitty regularization, 
