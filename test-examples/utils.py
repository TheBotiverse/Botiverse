import numpy as np
import copy
import matplotlib.pyplot as plt

def illustrate_features_2D(X, y, clf):
    '''
    Show 2D plot with decision regions
    '''
    # Fit the mode
    #X = X[:, :2]
    #clf = copy.copy(clf)
    #clf.fit(X, y)
    # Prepare the x,y grid so it spans all points
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1x1, x2x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
    
    # Predict for all points in the grid
    Z = clf.predict(np.c_[x1x1.ravel(), x2x2.ravel()]) 
    Z = Z.reshape(x1x1.shape)
    
    # Draw filled contours for the predictions
    plt.style.use('dark_background')
    plt.rcParams['figure.dpi'] = 150
    ax = plt.figure().add_subplot(111)
    ax.contourf(x1x1, x2x2, Z, alpha=0.4, cmap=plt.cm.Spectral)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
    plt.show()

