#!/usr/bin/env python
# coding: utf-8

# This reads in time-series photometry and does a PCA reduction of the brightest stars

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pytransit import QuadraticModel

time_series = pd.read_csv("test_time_series_20210917.csv", index_col=0)


# option to inject a transit
__ in corporate real JD time below; use time_series["jd"]__
length_dataset = len(time_series)
# time; don't know what form this will take
abcissa_time = np.linspace(0,1.2,length_dataset)
# generate transit model
tm = QuadraticModel()
#tm.set_data(times)
tm.set_data(abcissa_time)
# note t0 is time-of-center
# (k=0.1 gives transit depth of approx 1%)
model_transit = tm.evaluate(k=0.1, ldc=[0.2, 0.1], t0=0.6, p=0.7, a=3.0, i=0.5*np.pi)
noisy_transit = np.multiply(time_series["121"],model_transit)
transit_visual = np.multiply(0.5*np.median(time_series["121"]),model_transit)

import ipdb; ipdb.set_trace()

plt.plot(abcissa_time,time_series["121"],label="original")
plt.plot(abcissa_time,noisy_transit,label="transit")
plt.plot(abcissa_time,transit_visual, linestyle="--", color="k")
plt.legend()
plt.show()
import ipdb; ipdb.set_trace()

# PCA function partly cribbed from https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8
def pca(X):
    # X: data matrix, assumes 0-centered
    # N: number of PCA basis vectors to project data onto
    n, m = X.shape
    assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n-1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    print(eigen_vals.shape)
    # Project X onto PC space; these are the eigenvectors of the data
    X_pca = np.dot(X, eigen_vecs)

    ## BEGIN TEST
    print(eigen_vecs.shape)
    print(eigen_vecs)
    X_proj = np.dot(X[:,0],eigen_vecs)
    plt.plot(X_proj)
    plt.show()
    ## END TEST

    # determine cumulative explained variance
    variance_explained = []
    for i in eigen_vals:
        variance_explained.append((i/sum(eigen_vals)))

    cumulative_variance_explained = np.cumsum(variance_explained)

    # return orthonormal basis set, an array of cumulative variance, and eigenvalues of correlation matrix
    return X_pca, cumulative_variance_explained, eigen_vals


# project the time-series of bright stars onto eigenvectors, and return the projection and the explained variance
x_pca, variance_expl, e_vals = pca(x_scaled)


# project one star onto basis set


# return number of components that explain X amount of variance
# (note number can be a decimal)

def num_comps_var(variance_expl,variance_target):

    # abcissa (number of components); start with 1
    abcissa = np.add(1.,np.arange(len(variance_expl)))

    # interpolate (note a flipping of axes is necessary)
    comps_interp = np.interp(x=variance_target,xp=variance_expl,fp=abcissa)

    return comps_interp


# make list of brightest stars, with option to plot photometry,
# with annotations to identify stars

list_brightest = [] # initialize list of brightest stars
for (columnName, columnData) in time_series.iteritems():
    #print('Column Name : ', columnName)
    if np.median(columnData.values) > 5000:
        plt.plot(columnData.values)
        plt.annotate(str(columnName), xy=(2300,np.median(columnData.values)), xycoords="data")
        list_brightest.append(str(columnName))
'''
plt.xlabel("Frame number (~5 hr observation duration)")
plt.ylabel("Direct counts (no sky subtraction)")
plt.show()
'''


# option to remove any stars that may be intrinsically variable

#list_brightest.remove("25")
#list_brightest.remove("271")


# select the bright stars we want, and whiten the data
'''
# separate out the photometry from the brightest stars
x = time_series.loc[:, list_brightest].values

# standardize the photometry
x_scaled = StandardScaler().fit_transform(x)


# project the time-series of bright stars onto eigenvectors, and return the projection and the explained variance

x_pca, variance_expl = pca(x_scaled)


# subtract these first components from the data

photometry_decorr = np.subtract(x_scaled[:,0],np.dot(x_pca[:,:10],x_scaled[:,0]))
'''

# np.dot(x_scaled[:,0],x_pca[:,:10].T)


# plot explained variance

'''
plt.plot(variance_expl)
plt.show()
'''

'''
# plot eigenvectors

for i in range(0,len(x_pca[0,:])):
    plt.plot(x_pca[:,i])
plt.show()


plt.plot(single_star_data, label="empirical, post-scaling")
plt.show()

plt.plot(recon_light_curve,label="reconstructed")
plt.show()

plt.plot(photometry_decorr)
plt.show()
'''
