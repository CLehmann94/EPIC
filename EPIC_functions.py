#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 12:12:14 2021

@author: christian
"""

from astropy.io import fits
import math
import numpy as np


def A_Li(a, b1, b2, b3, c, T_eff, EW_Li):
    return np.log((EW_Li - a*T_eff - c) / b1) / b2 + b3


def A_Li_err(a, b2, c, T_eff, EW_Li, EW_Li_err):
    return EW_Li_err / np.abs(b2 * (EW_Li - a*T_eff - c))


def dMetal(data, a, b, c1, c2, d):
    T, logg, M = data[0] - 5772, data[1] - 4.438, data[2] - 0.0000
    return a*T + b*logg + c1*M + c2*M*M + d


def dTemp_lin(data, a1, b, c, c2, d):
    T, logg, M = data[0] - 5772, data[1] - 4.438, data[2] - 0.0000
    return a1*T + b*logg + c*M + c2*M*M + d


def dlogg_lin(data, a, b1, c, c2, d):
    T, logg, M = data[0] - 5772, data[1] - 4.438, data[2] - 0.0000
    return a*T + b1*logg + c*M + c2*M*M + d


# A number of simple functions used for the EPIC algorithm
def Gauss(x, a, x0, sigma):
    return 1 - a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def parabel(x, a, b, c):
    return a * (x - b)**2 + c


def hypersurface(B, data):
    x, y, z = B[0], B[1], B[2]
    a1, a2 = data[0], data[1]
    b1, b2 = data[2], data[3]
    c1, c2 = data[4], data[5]
    d1 = data[6]
    e = data[7]
    return a1*x + a2*x*x + b1*y + b2*y*y + c1*z + c2*z*z + d1*z/x + e


def hypersurfacelstsq(data, x, y, z):
    a1, a2 = data[0], data[1]
    b1, b2 = data[2], data[3]
    c1, c2 = data[4], data[5]
    d1 = data[6]
    e = data[7]
    return a1*x + a2*x*x + b1*y + b2*y*y + c1*z + c2*z*z + d1*z/x + e


def hypererr(B, data):
    x, y, z = B[0], B[1], B[2]
    a1, a2 = data[0], data[1]
    b1, b2 = data[2], data[3]
    c1, c2 = data[4], data[5]
    d1 = data[6]
    e = data[7]
    return (a1*x)**2 + (a2*x*x)**2 + (b1*y)**2 + (b2*y*y)**2 + \
           (c1*z)**2 + (c2*z*z)**2 + (d1*z/x)**2 + e**2


def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def linear_SNR(a, SNR_R):
    return a[0] + a[1] * SNR_R + a[2] * SNR_R * SNR_R


def read_sky(fitsfile):
    with fits.open(fitsfile) as hdulist:
        sky_w = np.array([el[0]*10 for el in hdulist[1].data])
        sky_f = [el[4] for el in hdulist[1].data]
        sky_f = np.multiply(1/np.max(sky_f), sky_f)
    return sky_w, sky_f


def find_nearest_idx(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or
       math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def pivot_to_ap(pivot):
    if pivot == 0:
        return 1
    fibre = pivot + (10 - 2 * int(str(pivot)[-1:]))
    if int(str(pivot)[-1:]) == 0:
        fibre = fibre - 20
    ap = fibre - int((pivot)/50)
    if ap == 392:
        ap = 391
    return ap


def renorm(f_ref):
    """Renormalizes spectral data by the maximum (only use on high SNR
    reference spectrum).
    Parameters
    ----------
        f_ref: Array like
            A subarray with flux values around a certain line
            for the reference spectrum.

    Returns
    -------
    A variable:
        raise_ref : Array like
            A number by which the spectrum must be multiplied

    """

    return 1 / np.max(f_ref)
