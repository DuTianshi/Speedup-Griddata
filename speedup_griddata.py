# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:36:28 2023

@author: DU
"""

import numpy as np
import scipy.spatial.qhull as qhull

def interp_weights(lon, lat, tlon, tlat, d=2):
    """
    mapping the relationship between the two axis

    Parameters
    ----------
    lon : float
        origin map x, 2 dimensional.
    lat : TYPE
        origin map y, 2 dimensional.
    tlon : TYPE
        new map x, 2 dimensional.
    tlat : TYPE
        new map y, 2 dimensional.
    d : int, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    TYPE
        returns the middle variable of griddata,
        use combine the function: interpolate.
    """
    xy = (lon.flatten(), lat.flatten())
    uv = (tlon.flatten(), tlat.flatten())
    xy = np.array(xy).T
    uv = np.array(uv).T
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts, shape):
    """
    using the mapping relationship on data

    Parameters
    ----------
    values : float
        the original data, 2 dimensional.
    vtx, wts : TYPE
        middle variable of griddata, obtain from the function: interp_weights.
    shape : 
        the shape of new data you need, because griddata need to flatten the 
        original data before use, thus the return data is one-dimensional, 
        this parameters is for reshape of the new data.

    Returns
    -------
    xx : float
        the interpolated data.

    """
    values = values.flatten()
    data = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    xx = data.reshape(shape)
    return xx
