#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:38:14 2021

@author: christian
"""
from astropy import constants as const
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve
import datetime as dt
import math
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats
from spectres import spectres
from tqdm import tqdm
import unyt as u


def add_weight(line_pos, line_wid, w, err, pix_wid):
    """Lines up the two spectra by the amount of light absorpted in the area
    around the line.
    Parameters
    ----------
        line_pos : float
            The position of the absorption line.
        line_wid : float
            The width of the absorption line.
        w : Array like
            A subarray with wavelength values around the line.
        err : Array like
            The corresponding error array.
        pix_wid : float
            The width of a pixel in wavelength.


    Returns
    -------
    Two variable:
        weight : Array like
            An array that weights the corresponding flux values for the
            wavelength array w.

    """
    i = 0
    j = -1
    npix = len(w)
# Initially every pixel is weighted by their inverse variance
    weight = np.divide(np.ones(len(w)), np.square(err))

# Pixel at a lower wavelength than the specified window have weight = 0
    while w[i] + pix_wid / 2 < line_pos - line_wid:
        weight[i] = 0.0
        i += 1
        npix -= 1

# Pixel at a higher wavelength than the specified window have weight = 0
    while w[j] - pix_wid / 2 > line_pos + line_wid:
        weight[j] = 0.0
        j -= 1
        npix -= 1

# The pixels on the edge of the window have a reduced weight according to
# their width within the window.
    weight[i] = weight[i] * (w[i] + pix_wid / 2 -
                             line_pos + line_wid) / pix_wid
    weight[j] = weight[j] * (pix_wid / 2 +
                             line_pos + line_wid - w[j]) / pix_wid

# Number of  pixels within the window takes into account fractions of pixels
    npix = npix - 2.0 + (pix_wid / 2 + line_pos + line_wid - w[j]) / \
        pix_wid + (w[i] + pix_wid / 2 - line_pos + line_wid) / pix_wid

# Normalising the weight by the heighest weight
    weight = np.divide(weight, max(weight))

    return weight, npix


def addSN(flux, time, vmag, DarkN, SkyN, n, norm_f, Boff=0.654, Roff=-0.352,
          Ioff=-0.7, HARSN=1000, HAR=False):
    """Adds noice to the inserted flux. The noise is dependent on the
    brightness of the target, the observation time, the dark noice and the
    sky noice. It simulates noice for a solar star. This simulates noise for
    a HERMES spectrum according to the capabilities of the spectrograph and
    telescope.
    Parameters
    ----------
        flux : Array like
            An array holding the flux.
        time : float
            Observing time (s).
        vmag : float
            Brightness in the V band (mag).
        DarkN : float
            Dark noise total photon count.
        SkyN : float
            Relative sky brightness.
        n : int
            Band identifier (0: B, 1: V, 2: R, 3: IR).
        norm_f : Array like
            Normalised flux array.
        Boff : float
            B band offset from V band (mag). Solar offset by default.
        Roff : float
            R band offset from V band (mag). Solar offset by default.
        Ioff : float
            IR band offset from V band (mag). Solar offset by default.
        HARSN : float
            Previous SNR in the original HARPS spectrum.
            (negligible by default)
        HAR : Boolean
            Has been a HARPS spectrum before. Will take into account previous
            noise of spectrum.

    Returns
    -------
    A variable:
        results : Library
            Contains:
                'SN' keyword for the resulting SN as a float
                'SNpp' keyword for SN per pixel as a float
                'e' keyword for the error numpy array
                'f' keyword for the new flux array
    """
    results = {}
# Determine the number of electrons observed in the specified band
    if n == 0:
        ne = time / 3600 * 10**(-0.4 * (0.993 * (vmag + Boff) - 24.05))
        nepp = ne / 3.81  # number of measured electrons per pixel
# Find the SNR of the initial HARPS spectrum for the wavelength region.
# Increases SNR per pixel for HERMES cause of larger pixels
        try:
            harSN = min(HARSN[31:36]) * 2
        except TypeError:
            harSN = HARSN * 2
        harSNpp = harSN / 3.81  # HARPS SNR per HERMES pixel
    elif n == 1:
        ne = time / 3600 * 10**(-0.4*(1.18 * vmag - 26.25))
        nepp = ne / 4.69
        try:
            harSN = min(HARSN[52:56]) * 2
        except TypeError:
            harSN = HARSN * 2
        harSNpp = harSN / 4.69
    elif n == 2:
        ne = time / 3600 * 10**(-0.4*(1.07 * (vmag + Roff) - 24.98))
        nepp = ne / 3.74
        try:
            harSN = min(HARSN[66:70]) * 2
        except TypeError:
            harSN = HARSN * 2
        harSNpp = harSN / 3.74
    elif n == 3:
        ne = time / 3600 * 10**(-0.4*(0.89 * (vmag + Ioff) - 22.33))
        nepp = ne / 3.74
        harSN = HARSN * 2
        harSNpp = harSN / 3.74

# Calculate the SNR (and SNR per pixel) and the number of sky pixel.
    skypp = SkyN * nepp * pow(2.5, vmag-17.552)
    SN = np.sqrt(ne)
    SNpp = math.sqrt(nepp + skypp)

# Compute results for HARPS spectra (calculate individual uncertainties and
# add random noise to the spectrum)
    if HAR:
        if harSN < SN:
            results['e'] = np.abs(np.divide(flux,
                                  np.sqrt(np.abs(norm_f))) / harSNpp)
            results['f'] = flux + DarkN * flux / ne
            results['SN'] = harSN
            results['SNpp'] = harSNpp
        else:
            SNadd = 1/math.sqrt(1/(SNpp**2) + 1/(harSNpp**2))
            adderr = flux / SNadd
            results['f'] = np.add(flux, np.random.normal(0, adderr,
                                  len(flux))) + DarkN * flux / ne
            results['e'] = np.abs(np.divide(flux,
                                            np.sqrt(np.abs(norm_f))) / SNpp)
            results['SN'] = SN
            results['SNpp'] = SNpp
# Compute results for HERMES spectra (calculate individual uncertainties and
# add random noise to the spectrum)
    else:
        results['SN'] = SN
        results['SNpp'] = SNpp
        results['e'] = np.abs(np.divide(flux, np.sqrt(np.abs(norm_f))) / SNpp)
        results['f'] = np.add(flux, np.random.normal(0, results['e'],
                                                     len(flux)))

    print(max(np.divide(results['f'], results['e'])))

    return results


def addSN_simple(flux, SNR, norm_f):
    """Adds noice to the inserted flux. This is the most simple way to do it.
    We take only a SNR and project noise of the projected strength on the flux
    array.
    Parameters
    ----------
        flux : Array like
            An array holding the flux.
        SNR : float
            The signal to noice ratio that shall be simulated on the flux
            array.
        norm_f : Array like
            An array holding the normalised flux.

    Returns
    -------
    A variable:
        results: Library
            Contains:
            'SN' keyword for the resulting SN as a float, the
            'SNpp' keyword for SN per pixel as a float
            'e' keyword for the error numpy array
            'f' keyword for the new flux array
    """
    results = {}

# Calculate the flux uncertainties and apply random noise to the spectrum
    results['SN'] = SNR
    results['SNpp'] = SNR
    results['e'] = np.abs(np.divide(flux, np.sqrt(np.abs(norm_f))) / SNR)
    results['f'] = np.add(flux, np.random.normal(0, results['e'], len(flux)))

    return results


def air_indexEdlen53(l, t=15., p=760.):
    """Return the index of refraction of air at given temperature, pressure,
    and wavelength in Angstroms.
    The formula is from Edlen 1953, provided directly by ESO.
    Parameters
    ----------
    l : float
        Vacuum wavelength in Angstroms
    t : float
        Temperature in °C. (Don't actually change this from the default!)
    p : float
        Pressure in mmHg. (Don't actually change this from the default!)
    Returns
    -------
    n : float
        The index of refraction for air at the given parameters.
    """

    n = 1e-6 * p * (1 + (1.049-0.0157*t)*1e-6*p) / 720.883 / (1 + 0.003661*t)\
        * (64.328 + 29498.1/(146-(1e4/l)**2) + 255.4/(41-(1e4/l)**2))
    n = n + 1
    return n


def air2vacESO(air_wavelengths_array):
    """Take an array of air wavelengths and return an array of vacuum
    wavelengths in the same units.
    Parameters
    ----------
    air_arr : `unyt.unyt_array`
        A list of wavelengths in air, with dimensions length. Will be converted
        to Angstroms internally.
    Returns
    -------
    `unyt.unyt_array`
        A unyt_array of wavelengths in vacuum, in the original units.
    """

    reshape = False
    original_units = air_wavelengths_array.units
    if air_wavelengths_array.ndim == 2:
        # We need to flatten the array to 1-D, then reshape it afterwards.
        reshape = True
        original_shape = air_wavelengths_array.shape
        tqdm.write(str(original_shape))
        air_wavelengths_array = air_wavelengths_array.flatten()

    air_wavelengths_array.convert_to_units(u.angstrom)

    tolerance = 2e-12
    num_iter = 100

    vacuum_wavelengths_list = []

#    tqdm.write('Converting air wavelengths to vacuum using Edlen 1953.')
    for i in range(0, len(air_wavelengths_array)):
        new_wavelength = air_wavelengths_array[i].value
        old_wavelength = 0.
        iterations = 0
        past_iterations = [new_wavelength]
        while abs(old_wavelength - new_wavelength) > tolerance:
            old_wavelength = new_wavelength
            n_refraction = air_indexEdlen53(new_wavelength)
            new_wavelength = air_wavelengths_array[i].value * n_refraction
            iterations += 1
            past_iterations.append(new_wavelength)
            if iterations > num_iter:
                print(past_iterations)
                raise RuntimeError('Max number of iterations exceeded!')

        vacuum_wavelengths_list.append(new_wavelength)
    vacuum_array = u.unyt_array(vacuum_wavelengths_list, u.angstrom)

    if reshape:
        tqdm.write(f'Converting back to original shape: {original_shape}.')
        # Reshape the array back to its original shape.
        return vacuum_array.reshape(original_shape).to(original_units)
    else:
        return vacuum_array.to(original_units)


def center_line(w, f):
    """Measures the center of an absorption line.
    Parameters
    ----------
        w : Array like
            A subarray with wavelenghts within a certain line.
        f : Array like
            A subarray with flux values within a certain line.
        deg : int
            The degree of the fitting polynomial.
        band : int
            The band in which the line is within the HERMES spectrograph.

    Returns
    -------
    A variable:
        x_min : float
            The wavelength value of the line after centering on the minimum.
    """
    a = np.argmin(f)
    w_pix = w[a] - w[a-1]
    f2 = f[a-1:a+2]
    A = np.array([[1, -1, 1], [0, 0, 1], [1, 1, 1]])
    B = np.array([f2[0], f2[1], f2[2]])
    X = np.linalg.inv(A).dot(B)

    w_npix = - X[1] / (2*X[0])

    if np.absolute(w_npix) > 1:
        print("Error: Minimum not close to minimum")
    if X[0] < 0:
        print("Error: Minimum is actually maximum")


#   Return the center of the Gaussian
    return w[a] + (w_npix * w_pix)


def Gauss(x, a, x0, sigma):
    return 1 - a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def determine_resolving_power(w, f, deg=2, band=0, specres=28000, w2=[],
                              f2=[]):
    """Determines the resolving power of an absorption feature.
    Parameters
    ----------
        w: Array like
            A subarray with wavelenghts within a certain line.
        f: Array like
            A subarray with flux values within a certain line.
        deg: int
            The degree of the fitting polynomial.
        band: int
            The band in which the line is within the HERMES spectrograph

    Returns
    -------
    A variable:
        x_min : float
            The wavelength value of the line after centering on the minimum
    """
#   calculate the sigma of the band
    boun = [[4715, 4900], [5649, 5873], [6478, 6737], [7585, 7885]]
    c = const.c.to('km/s')
    sig_final = 0

    sig = (boun[band][1] + boun[band][0]) / (2.0 * 2.355 * specres)

#   Fit a Gaussian to the line
    mean = sum(w * f) / sum(f)
    try:
        popt, pcov = curve_fit(Gauss, w, f, p0=[1-min(f), mean, sig])
    except RuntimeError:
        return -1, -1, -1

    w_plot = np.linspace(w[0], w[-1], 100)
    plt.step(w_plot, Gauss(w_plot, popt[0], popt[1], popt[2]))
    plt.step(w, f)
    plt.show()
    plt.clf()

    if 0.1 < popt[0] and popt[0] < 0.7:
        sig2 = popt[2]
        sig2_err = np.sqrt(np.diag(pcov))[2]
        R = float(popt[1] / (2.355 * popt[2]))
#        try:
#            def Gauss2(x, a, b, sigma, c):
#                return b - a * np.exp(-(x - c)**2 / (2 * sigma**2))
#            popt2, pcov2 = curve_fit(Gauss2, w, f, p0=[1-min(f), 1.0, sig,
#                                                       popt[1]])
#            R = float(popt2[3] / (2.355 * popt2[2]))
#        except RuntimeError:
#            return -1, -1, -1
    else:
        return -1, -1, -1
    sig_b = np.square(sig2) - np.square(sig)
    if sig_b < 0:
        sig_abs = np.sqrt(np.abs(sig_b))
        sig_err = np.sqrt(np.square(sig2 * sig2_err * c.value /
                                    (popt[1] * sig_abs)) +
                          np.square(sig_abs * c.value / popt[1]**2))
        sig_final = 0
    elif sig_b >= 0:
        sig_b2 = np.sqrt(sig_b)
        sig_err = np.sqrt(np.square(sig2 * sig2_err * c.value /
                                    (popt[1] * sig_b2)) +
                          np.square(sig_b2 * c.value / popt[1]**2))
        sig_final = sig_b2 * c.value / popt[1]

    return R, sig_final, sig_err


def determine_radvel(ref_flux, tar_flux, pixel, rv_weight, mpix=0,
                     plot_correlation=False, band=0,
                     mid_wav=0.0, secondary=False):
    """Determine the radial velocity between two Spectra
    Parameters
    ----------
        ref_flux : Array like
            A subarray holding the reference's flux.
        tar_flux : Array like
            A subarray holding the targets flux.
        pixel : float
            The width of a pixel to determine the total shift in wavelength.
        rv_weight : Array like
            An array with  the weigths of all pixels within this correction.
        mpix : float
            Eliminates pixels from the process as a fraction of total pixels.
            E.g. 0.2 will eliminate 20% of pixels from both edges of the
            spectrum (whis will reduce the number of pixel by 40%).
        plot_correlation : Boolean
            If True, plots the correlation function. Mainly used for
            debugging.
        band : Int
            Querries the band in which this correction happens. This is only
            relevant for the IR band (band=3) for which the algorithm needs
            to be cautious because of sky correction residuals.
        mid_wav : float
            The middle of the correction to determine the radial velocity in
            km/s.
        secondary : Boolean
            If True, the correction is in a more precise correction mode to
            correct only small radial velocities (<8 HERMES pixel).
    Returns
    -------
    A variable:
        shift : float
            The radial velocity between the spectra.
    """

    max_pix = 20
    c = const.c.to('km/s')
    tar_flux = np.array(tar_flux)
    if band == 3:
        tar_flux[tar_flux > 0.5] = np.ones_like(tar_flux[tar_flux > 0.5]) * 0.5

    if mpix > 0.0 and mpix < 0.3:
        pix_elim = int(len(ref_flux)*mpix)
        ref_flux = ref_flux[pix_elim:-pix_elim]

    corr = np.array([])
    rv_weight = np.array(rv_weight)
    rv_weight = np.where([f < 0 for f in tar_flux], 0.0, rv_weight)

    if all(rv_weight == 0):
        rv_weight = np.ones_like(rv_weight)

    k = 0
    while len(ref_flux)+k <= len(tar_flux):
        weight = rv_weight[k:len(ref_flux)+k]
        corr_temp = np.divide(
                np.sum(np.multiply(np.multiply(
                        ref_flux, tar_flux[k:len(ref_flux)+k]), weight)),
                np.multiply(np.sqrt(np.sum(np.multiply(
                        weight, np.square(ref_flux)))),
                        np.sqrt(np.sum(np.multiply(
                                weight, np.square(
                                        tar_flux[k:len(ref_flux)+k]))))))
        corr = np.append(corr, corr_temp)

        k = k+1

    pix_zero = int(len(corr) / 2)

    if plot_correlation is True:
        plt.plot(range(len(corr)), corr)
        plt.show()
        plt.close()
    if secondary is True:
        min_i = np.argmax(corr[pix_zero-max_pix:pix_zero+max_pix]) + \
            pix_zero-max_pix
    else:
        min_i = np.argmax(corr)

    shift = (min_i - pix_zero) * pixel

    if plot_correlation is True:
        plt.plot(range(len(ref_flux)) + min_i, 1 - ref_flux)
        plt.plot(range(len(tar_flux)), 1 - tar_flux)
        plt.show()
        plt.close()

    if mid_wav != 0:
        corr_range = np.linspace(0, len(corr) - 1, len(corr))
        corr_rv = (corr_range - pix_zero) * pixel * c.value / mid_wav
        return shift, corr_rv, corr
    else:
        return shift


def prepare_reference_rv(wave_r_old, flux_r_old, wave_t, res_power, center_w,
                         stacked=False, single_out=True, harps=False,
                         test=False):
    """Convolves and resamples a high resolution reference spectrum onto the
    target wavelength grid.
    Parameters
    ----------
        wave_r_old : Array like
            The old reference wavelength array.
        flux_r_old : Array like
            The old reference flux array.
        wave_t : Array like
            The wavelength array on which the spectrum will be projected on.
        res_power : Array like
            The resolving power of the target spectrum.
        center_w : float
            Wavelength at the centre of the array. Used to determine the
            width of the convolving gaussian from the resolving power.
        stacked : Boolean
            If True, assumes the spectrum to be a stacked spectrum or to have
            a "generic" resolving power layout as a result of being convolved
            with HERMES resolving power.
        single_out : Boolean
            If True, doesn't return f_conv or w_temp (True by default).
        harps : Boolean
            Recognises HARPS spectra and takes into account their limited
            resolving power.
    Returns
    -------
    Variable:
        flux_r : Numpy Array
            The resulting reference flux array that fits the target wavelength
            array.
        f_conv : Numpy Array
            The flux array on the old wavelength grid.
        w_temp : Numpy Array
            The old wavelength grid.
    """

    if stacked is True:
        band_correction = [0.8, 0.775, 0.75, 0.83]
        j = int(wave_t[0] / 1000 - 4)
        res_power = 28000 * band_correction[j]

    if harps is True:
        harps_res = 75000
        res_power = 1 / np.sqrt(
                1/np.square(res_power) - 1/np.square(harps_res))

    w_temp = wave_r_old[np.bitwise_and(wave_t[0] - 10 < wave_r_old,
                                       wave_r_old < wave_t[-1] + 10)]
    f_temp = flux_r_old[np.bitwise_and(wave_t[0] - 10 < wave_r_old,
                                       wave_r_old < wave_t[-1] + 10)]
    w_pix = w_temp[1] - w_temp[0]

    sigma = center_w / (2.355 * res_power * w_pix)

    if test is True:
        c = const.c.to('km/s')
        lw2_wav = w_temp[0] * 15.0 / c.value / w_pix
        sigma2 = w_temp[int(len(w_temp)/2)] / (2.355 * 28000 * w_pix)
        mu = w_temp[0]
        x = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)
        y = np.linspace(mu - lw2_wav, mu + lw2_wav, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))
        plt.plot(x, stats.norm.pdf(x, mu, sigma2))
        plt.axvline(mu + lw2_wav, color='black')
        plt.axvline(mu - lw2_wav, color='black')
        print(np.sum(stats.norm.pdf(y, mu, sigma)) /
              np.sum(stats.norm.pdf(y, mu, sigma2)))
        plt.show()

    Gauss = Gaussian1DKernel(stddev=sigma)
    f_conv = convolve(f_temp, Gauss)

    flux_r = spectres(wave_t, w_temp, f_conv)

    if test is True:
        print(sigma)
        plt.plot(wave_t, flux_r)
        plt.plot(w_temp, f_temp)
        plt.show()
        plt.clf()

    if single_out is True:
        return flux_r
    else:
        return flux_r, f_conv, w_temp


def prepare_reference(wave_r_old, flux_r_old, res_power,
                      stacked=False):
    """Convolves and resamples a high resolution reference spectrum onto the
    target wavelength grid.
    Parameters
    ----------
        wave_r_old : Array like
            The old reference wavelength array.
        flux_r_old : Array like
            The old reference flux array.
        wave_t : Array like
            The wavelength array on which the spectrum will be projected on.
        res_power : Array like
            The resolving power of the target spectrum.
        stacked : Boolean
            If True, assumes the spectrum to be a stacked spectrum or to have
            a "generic" resolving power layout as a result of being convolved
            with HERMES resolving power.
    Returns
    -------
    Variable:
        flux_r : Array like
            The resulting reference flux array that fits the target wavelength
            array.

    """

    if stacked is True:
        band_correction = [0.8, 0.775, 0.75, 0.83]
        j = int(wave_r_old[0] / 1000 - 4)
        res_power = 28000 * band_correction[j]

    w_temp = wave_r_old
    f_temp = flux_r_old
    w_pix = w_temp[1] - w_temp[0]

    sigma = w_temp[int(len(w_temp)/2)] / (2.355 * res_power * w_pix)

    Gauss = Gaussian1DKernel(stddev=sigma)
    f_conv = convolve(f_temp, Gauss)

    return f_conv


def lineup(f_ref, f_tar, e_ref, e_tar, band=0, low_perc=False,
           rv_weight=[0], Li_plot=False):
    """Lines up the two spectra by the amount of light absorpted in the
    area around the line.
    Parameters
    ----------
        f_ref: Array like
            A subarray with flux values around a certain line for the
            reference spectrum.
        f_tar: Array like
            A subarray with flux values around a certain line for the
            target spectrum.
        e_ref: Array like
            A subarray with error values around a certain line for the
            reference spectrum.
        e_tar: Array like
            A subarray with error values around a certain line for the
            target spectrum.
        band : int
            The band in which we want to use this algorithm.
            (0: B, 1: V, 2: R, 3: IR)
        low_perc : Boolean
            If True, ignores the lowest 75% of flux values in the reference
            spectrum and the corresponding pixel in the target.
        rv_weight : Array like
            Gives relative weights to all pixels. Note that the aaray length
            must be the same as the length of f_ref and e_ref.

    Returns
    -------
    A variable:
        raise_tar : Array like
            A number by which the target spectrum is multiplied in order
            to line it up with the reference.

    """
    if Li_plot is True:
        plt.step(np.linspace(0, len(f_ref)-1, len(f_ref)), f_ref)
        plt.step(np.linspace(0, len(f_tar)-1, len(f_tar)), f_tar)
        plt.axvline(len(f_ref)/2)
        plt.show()
        plt.clf()
    i = band
    perc = 1.0
    if low_perc is True:
        perc = 0.25
    b_coeff = [3, 3, 3, 3]
    b_coeff2 = [5, 5, 5, 5]
    if all(rv_weight == 0) or len(rv_weight) != len(e_tar):
        rv_weight = np.ones_like(e_tar)
    weight = 1 / np.square(e_tar) * rv_weight

    cut_value = np.sort(f_ref)[int(len(f_ref)*(1-perc))]
    f_tar = f_tar[f_ref > cut_value]
    weight = weight[f_ref > cut_value]
#    weight = np.ones_like(weight[f_ref > cut_value])
    f_ref = f_ref[f_ref > cut_value]

    sum1 = sum(f_tar * weight)

    if sum1 == 0:
        return False

    raise_tar = sum(f_ref * weight) / sum1

    m_flag = False
    for j in range(4):
        f_tar_new = f_tar * raise_tar
        e_tar_new = e_tar * raise_tar
        con = f_tar_new < np.max(
                [1.05 + e_tar_new * b_coeff[i],
                 1.0 + e_tar_new * b_coeff2[i]])
        if np.median(f_tar_new[con]) > 1.05 or m_flag is True:
            con = np.bitwise_and(con, f_tar_new > np.min(
                    [0.9 - e_tar_new * b_coeff[i],
                     1.0 - e_tar_new * b_coeff2[i]]))
            m_flag = True

        f_ref_new = f_ref[con]
        weight_new = weight[con]
        f_tar_new = f_tar_new[con]

        raise_tar = raise_tar * sum(f_ref_new * weight_new) / \
            sum(f_tar_new * weight_new)

    return raise_tar


def line_prep_plot(center_w, center_f, linew, linew_old, window,
                   post_resolv_w, post_resolv_f, target_resolv_w,
                   target_resolv_f, post_norm_w, post_norm_f, reference_norm_w,
                   reference_norm_f, weights, twavcon):
    """
    Makes a plot for section 2 of the Lehmann et al. 2021 paper.

    """
    c = const.c.to('km/s')
    l1_vel = 15
    l2_vel = 400

    lower_bound = linew - window*1.4
    upper_bound = linew + window

    pre_resolv_w, pre_resolv_f = center_w, center_f

    pre_norm_w, pre_norm_f = target_resolv_w, target_resolv_f

    reference_EW_w, reference_EW_f, target_EW_w, target_EW_f = \
        reference_norm_w, reference_norm_f, post_norm_w, post_norm_f

    l1_wav = linew * l1_vel / c.value
    l2_wav = linew * l2_vel / c.value

    output1 = 'Paper_Prep_plot1.pdf'
    output2 = 'Paper_Prep_plot2.pdf'
    output3 = 'Paper_Prep_plot3.pdf'
    output4 = 'Paper_Prep_plot4.pdf'
    output5 = 'MCR_spectrum.pdf'

    pdf1 = matplotlib.backends.backend_pdf.PdfPages(output1)
    pdf2 = matplotlib.backends.backend_pdf.PdfPages(output2)
    pdf3 = matplotlib.backends.backend_pdf.PdfPages(output3)
    pdf4 = matplotlib.backends.backend_pdf.PdfPages(output4)
    pdf5 = matplotlib.backends.backend_pdf.PdfPages(output5)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    fig.set_size_inches(8.5, 4.5)
    ax.step(center_w, center_f, lw=4, where='mid', color='blue',
            label='Reference spectrum')
    ax.axhline(1, color='black', ls='--', lw=4)
    ax.axvline(linew, color='black', ls='-', lw=4, label='Reference Centroid')
    ax.axvline(linew_old-0.02, color='black', ls='dotted', lw=4,
               label='Line list wavelength')
    ax.set_xlim(lower_bound, upper_bound)
    ax.set_ylim(0.4, 1.05)
#    ax.set_xlabel(r'\LARGE Wavelength [\AA]')
    ax.set_ylabel(r'Normalized flux')
    ax.legend(loc='lower left', handlelength=1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_rasterization_zorder(-10)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98,
                        wspace=0.0, hspace=0.0)
    pdf1.savefig(fig, bbox_inches='tight', pad_inches=0.02)
    pdf1.close()
    ax.clear()

    ax.step(pre_resolv_w, pre_resolv_f, lw=4, where='mid', color='blue',
            label='Unconvolved reference')
    ax.step(post_resolv_w, post_resolv_f, lw=4, where='mid', color='purple',
            label='Convolved reference')
    ax.step(target_resolv_w, target_resolv_f, lw=4, where='mid', color='red',
            label='HERMES target')
    ax.axhline(1, color='black', ls='--', lw=4)
    ax.axvline(linew, color='black', ls='-', lw=4)
    ax.set_xlim(lower_bound, upper_bound)
    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel(r'\LARGE Wavelength [\AA]')
    ax.set_ylabel(r'Normalized flux')
    ax.legend(loc='lower left', handlelength=1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_rasterization_zorder(-10)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98,
                        wspace=0.0, hspace=0.0)
    pdf2.savefig(fig, bbox_inches='tight', pad_inches=0.02)
    pdf2.close()
    ax.clear()

    ax.step(pre_norm_w, pre_norm_f-0.1, lw=4, where='mid', color='red',
            label='Pre-norm target')
    ax.step(post_norm_w, post_norm_f, lw=4, where='mid', color='orange',
            label='Post-norm target')
    ax.step(reference_norm_w, reference_norm_f, lw=4, where='mid',
            color='purple', label='Reference')
    ax.axhline(1, color='black', ls='--', lw=4)
    ax.axvline(linew, color='black', ls='-', lw=4)
    ax.set_xlim(linew - l2_wav, linew + l2_wav)
    ax.set_ylim(0.4, 1.05)
#    ax.set_xlabel(r'wavelength [\AA]')
    ax.set_ylabel(r'Normalized flux')
    ax.legend(loc='lower left', handlelength=1.0)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_rasterization_zorder(-10)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98,
                        wspace=0.0, hspace=0.0)
    pdf3.savefig(fig, bbox_inches='tight', pad_inches=0.02)
    pdf3.close()
    ax.clear()

    ax.step(reference_EW_w, reference_EW_f, lw=4, where='mid', color='purple',
            label='Reference')
    ax.step(target_EW_w, target_EW_f, lw=4, where='mid', color='orange',
            label='HERMES target')
    ax.step(target_EW_w[twavcon], weights/max(weights)/3 + 0.39, lw=4,
            where='mid', label="Weights")
    ax.axhline(1, color='black', ls='--', lw=4)
    ax.set_xlim(lower_bound, upper_bound)
    ax.axvline(linew, color='black', ls='-', lw=4)
    ax.axvline(linew - l1_wav, color='red', ls='--', lw=4)
    ax.axvline(linew + l1_wav, color='red', ls='--', lw=4)
    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel(r'Wavelength [\AA]')
    ax.set_ylabel(r'Normalized flux')
    ax.legend(loc='lower left', handlelength=1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_rasterization_zorder(-10)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98,
                        wspace=0.0, hspace=0.0)
    pdf4.savefig(fig, bbox_inches='tight', pad_inches=0.02)
    pdf4.close()
    ax.clear()

    ax.step(pre_resolv_w, pre_resolv_f, lw=2.5, where='mid',
            label='R>100,000')
    ax.step(post_resolv_w, post_resolv_f, lw=2.5, where='mid',
            label='R$\sim$28,000')
    ax.axhline(1, color='black', ls='--', lw=2.5)
    ax.set_xlim(linew - l2_wav-0.7, linew + l2_wav-8)
    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel(r'Wavelength [\AA]')
    ax.set_ylabel(r'Normalized flux')
    ax.legend(loc='lower left', handlelength=1.0)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_rasterization_zorder(-10)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98,
                        wspace=0.0, hspace=0.0)
    pdf5.savefig(fig, bbox_inches='tight', pad_inches=0.02)
    pdf5.close()
    ax.clear()


def measure_EW(w, f, err, weight, line_wid):
    """Uses the weight and the given pixel to measure the EW
    Parameters
    ----------
        w : Array like
            A subarray with wavelength values around a certain
            line for the reference spectrum.
        f : Array like
            A subarray with flux values around a certain
            line for the reference spectrum.
        err: Array like
            The flux error array
        weight : Array like
            A subarray with weights for flux values around
            a certain line for the target spectrum.
        line_wid : float
            Width of the absorption line window in which to measure the EW.

    Returns
    -------
    Two variable:
        EW : float
            The equivalent width of the line.
        EW_sig : float
            The uncertanty of the equivalent width.

    """

    absorb = np.subtract(1.0, f)

    abs_bar = np.sum(np.multiply(absorb, weight)) / np.sum(weight)
    sig = np.sqrt(np.sum(np.multiply(np.square(weight), np.square(err)))
                  / np.square(np.sum(weight)))

    EW = abs_bar * line_wid * 1000
    EWs = sig * line_wid * 1000

    return EW, EWs


def prepare_target_wavelegnth(wave_new, wave_old, flux):
    """Re-samples a spectrum with a high pixelwidth to a spectrum with low
    pixelwidth. This will not increase the accuracy of the spectrum.
    Parameters
    ----------
        wave_new : Array like
            The new wavelength array on which the flux is projected.
        wave_old : Array like
            The old wavelenght array for the flux.
        flux : Array like
            The flux array of the spectrum.
    Returns
    -------
    Variable:
        flux_new : Array like
            The resulting flux array for the wave_new wavelength array.

    """

    flux_new = []
    wav_disp_new = (wave_new[-1] - wave_new[0]) / len(wave_new)
    wav_disp_old = (wave_old[-1] - wave_old[0]) / len(wave_old)

    if wav_disp_new > wav_disp_old:
        print('Error: Old wavelegnth array is finer than the new one.')
        return 1
    if wave_new[0] < wave_old[0] or wave_new[-1] > wave_old[-1]:
        print('Error: New wavelength array must be contained '
              + 'within the old one.')
        return 1

    for w_new in wave_new:
        i = np.argwhere(np.array(wave_old) < w_new)[-1][0]
        dist1, dist2 = w_new - wave_old[i], wave_old[i+1] - w_new
        w1, w2 = 1 - (dist1 / wav_disp_old), 1 - (dist2 / wav_disp_old)
        flux_new = np.append(flux_new, flux[i] * w1 + flux[i+1] * w2)

    return flux_new


def readlinelistw(linelistfile):
    """Read a linelist and return the lines Wavelenghts, element, ionisation
    and excitation potential.
    Parameters
    ----------
    linelistfile : str or Path object
        A path to a linelist file to be read.

    Returns
    -------
    Four seperate variables:
        w : Numpy array
            The wavelength values of all the lines.
        elem : Numpy array
            The element type of the corresponding lines.
        ion : Numpy array
            The ionisation of the corresponding lines.
        ep : Numpy array
            The excitation potential of the corresponding lines.
    """

#    print(linelistfile)
    with open(linelistfile) as linelist:
        lines = linelist.readlines()
        del_index = []
        w = np.zeros_like(lines, dtype=float)
        elem = np.chararray((len(lines),), unicode=True, itemsize=6)
        ion = np.ones_like(lines, dtype=int)
        ep = np.zeros_like(lines, dtype=float)

        for i, line in enumerate(lines):
            if line.startswith(';') or line.startswith('#'):
                del_index.append(i)
                continue
            line = line.replace("|", " ")
            w[i] = line.split()[1]
            elem[i] = line.split()[0]

        if all(ws <= 3 for ws in w):
            del_index = []
            ion = np.array(w)
            for i, line in enumerate(lines):
                if line.startswith(';') or line.startswith('#'):
                    del_index.append(i)
                    continue
                line = line.replace("|", " ")
                w[i] = float(line.split()[2])
                if len(line.split()) > 3:
                    ep[i] = float(line.split()[3])

    w = np.delete(w, del_index)
    elem = np.delete(elem, del_index)
    ion = np.delete(ion, del_index)
    ep = np.delete(ep, del_index)

    return w, elem, ion, ep


def read_ap_correct(w, elem, ion, ap_number, callibration, ap_weight, band=0):
    """Read the correction file for lines and return the correction values.
    Parameters
    ----------
    w: array-like
        Wavelength of the lines.
    elem: array like
        Element of the lines.
    ion: array-like
        Ionization of the lines.
    ap_numper : int or array like
        The aperture number in which the spectrum was observed. Can be array
        if spectrum is observed over multiple nights in different apertures.
    callibration : str or Path object
        The file in which the aperture scaling corrections are stored.
    ap_weight : array like
        Contains the weights for each contributing aperture.
        Normally this is weighted with the SNR of each spectrum for the
        combination process.
    Returns
    -------
    ap_correct: Numpy array
        Correction for each line. The array is in the same structure as the
        line arrays.

    """
    b_ex = [[4000, 8000], [4000, 5000], [5000, 6000], [6000, 7000],
            [7000, 8000]]
    ap_correct = np.ones_like(w)
    ap_correct2 = [[]] * len(ap_number)

    if ap_number[0] == 0:
        return ap_correct
    else:
        for j in range(len(ap_number)):
            with open(callibration + 'Resolving_map_correction/aperture' +
                      str(ap_number[j]) + '_correct.dat') as corr_file:
                lines = corr_file.readlines()
            if j == 0:
                w2 = np.zeros_like(lines, dtype=float)
                elem2 = np.chararray((len(lines),), unicode=True, itemsize=6)
                ion2 = np.ones_like(lines, dtype=int)
            ap_correct2[j] = np.ones_like(lines, dtype=float)

            for i, line in enumerate(lines):
                if line.startswith(';') or line.startswith('#'):
                    continue
                if float(line.split()[2]) < b_ex[band][0] \
                   or float(line.split()[2]) > b_ex[band][1]:
                    continue
                if j == 0:
                    elem2[i] = line.split()[0]
                    ion2[i] = int(line.split()[1])
                    w2[i] = float(line.split()[2])
                ap_correct2[j][i] = float(line.split()[3]) * ap_weight[j]

        ap_correct3 = np.sum(ap_correct2, axis=0)

        for i in range(len(w)):
            try:
                corr_index = np.where(np.bitwise_and(np.bitwise_and(
                        elem2 == elem[i], ion2 == ion[i]), w2 == w[i]))[0][0]
                ap_correct[i] = ap_correct3[corr_index]
            except IndexError:
                ap_correct[i] = 1.0

    return ap_correct


def combine_ap(combine_file, spec_name):
    """Find all apertures used in HERMES to combine a resulting aperture array.
    The weights of the spectrum are given by the SNR^2.
    Parameters
    ----------
    combine_file: str or Path object
        The name of the file which contains the neccessary information.
        Normally kept in the calibration folder.
    spec_name: str or Path object
        The identifier used in the table for this target

    Returns
    -------
    ap_array: Numpy array
        All aperture numbers that participated in the combined spectrum.
    weights: Numpy array
        The weight of all apertures that are part of the observation.

    """
    aperture_array = []
    snr_array = []
    with open(combine_file, 'r') as comb:
        lines = comb.readlines()
    for line in lines:
        if line.startswith('Name'):
            continue
        if line.startswith(spec_name):
            aperture_array = np.array(line.split(',')[3::5])
            snr_array = np.array(line.split(',')[5::5])
            snr_array = np.array(snr_array[aperture_array != 'NaN'],
                                 dtype='float')
            aperture_array = np.array(
                    np.array(aperture_array[aperture_array != 'NaN'],
                             dtype='float'), dtype='int')

    if len(aperture_array) == 0:
        return [], []

    weight_array = np.square(snr_array) / np.sum(np.square(snr_array))

    for i in range(len(aperture_array)):
        if len(aperture_array) < i+1:
            break
        indic = np.argwhere(aperture_array == aperture_array[i])
        weight_array[i] = np.sum(weight_array[indic])
        weight_array = np.delete(weight_array, indic[1:])
        aperture_array = np.delete(aperture_array, indic[1:])
        aperture_array = np.where([a == 0 for a in aperture_array],
                                  1, aperture_array)

    return aperture_array, weight_array


def rHERMES(FITSfile, datahdu=0, SN=False, e_hdu=1, plot_sky=False):
    """Read a HERMES FITS file and returns data information.
    Parameters
    ----------
    FITSfile : str or Path object
        A path to a HARPS FITS file to be read.
    datahdu  : int
        Decides which data hdulist to read the data from
        0 is Flux, 4 is normalized Flux in HERMES spectra.

    Returns
    -------
    dict
        A dictionary containing the following key-value pairs:
        w : Numpy array
            The wavelength array.
        f : Numpy array
            The flux array.
        e : Numpy array
            A zero array to be changed later
    """

    result = {}
    if FITSfile.endswith('.fits'):
        with fits.open(FITSfile) as hdulist:
            header0 = hdulist[0].header
            f = hdulist[datahdu].data
            unnorm_f = hdulist[0].data
            sky_f = hdulist[2].data
            e = hdulist[e_hdu].data
            cdelta1 = header0['CDELT1']
            crval1 = header0['CRVAL1']
            rv_weight = np.ones_like(f)
            for i in range(len(sky_f)):
                if sky_f[i] < 0:
                    rv_weight[i] = 0
#            create wavelength and error (only 0 values by this point) array
            w = np.linspace(0, len(f) - 1, len(f)) * cdelta1 + crval1
#            If we want to use the normalized spectrum, we should use a
#            normalized error
#            if datahdu == 4:
#                e = np.divide(np.multiply(e, f), hdulist[0].data)
#            write array on output
            result['w'] = w
            result['f'] = f
            result['e'] = e
            result['disp'] = w[1] - w[0]
            result['rv_weight'] = rv_weight

            if SN and 'SNR' in header0:
                SNR = header0['SNR']
                result['SNR'] = SNR
            else:
                result['SNR'] = 1000

            if plot_sky is True:
                print(np.subtract(sky_f, unnorm_f))
                fig, ax = plt.subplots(nrows=1, ncols=1)

                fig.set_size_inches(8.5, 4.5)
                ax.step(w, sky_f, lw=2, where='mid', color='blue',
                        label='with sky')
                ax.step(w, unnorm_f, lw=2, where='mid', color='red',
                        label='without sky')
                ax.axhline(1, color='black', ls='--', lw=4)
                ax.set_xlabel(r'\LARGE Wavelength [\AA]')
                ax.set_ylabel(r'Normalized flux')
                ax.legend(loc='lower left', handlelength=1.0)
                ax.set_rasterization_zorder(-10)
                plt.show()

        return result

    else:
        wave = np.array([], dtype='float64')
        flux = np.array([], dtype='float64')

        with open(FITSfile) as data:
            for line in data:
                if line.startswith('#'):
                    continue
                wave = np.append(wave, float(line.split(',')[0]))
                flux = np.abs(np.append(flux, float(line.split(',')[1])))

        result['w'] = wave
        result['f'] = flux
        result['e'] = np.absolute(np.divide(np.power(flux, 0.4), 1000))
        result['SNR'] = 1000

        return result


def r_resolving_map(FITSfile, ap_number, warn_flag=True, print_max_diff=False,
                    weight=[1]):
    """Read a HERMES FITS file containing a resolving power map as written in
    Kos et al 2016.
    Parameters
    ----------
    FITSfile : str or Path object
        A path to the FITS file to be read.
    ap_number: int
        Number of aparature used for the target spectrum.

    Returns
    -------
    dict
        A dictionary containing the following key-value pairs:
        w : Numpy array
            The wavelength array.
        R : Numpy array
            Array of resolving powers in all querried lines.


    """
    R = [[]] * len(ap_number)
    with fits.open(FITSfile, mode='readonly') as hdulist:
        if print_max_diff is True:
            stuff = hdulist[0].data[hdulist[0].data > 0]
            print(len(stuff) / len(hdulist[0].data[0]))
            stuff = np.reshape(stuff, (int(len(stuff) /
                                       len(hdulist[0].data[0])),
                                       len(hdulist[0].data[0])))
            print(np.amax(np.subtract(np.amax(stuff, axis=0),
                                      np.amin(stuff, axis=0))))
        for i in range(len(ap_number)):
            header0 = hdulist[0].header
            R[i] = np.multiply(hdulist[0].data[ap_number[i]], weight[i])
            for r in R[i]:
                if (r < 10000 and len(ap_number) == 1) or r == 0:
                    if warn_flag is True:
                        print('Warning: Aperture does not contain resolving' +
                              ' power.')
                    R[i] = np.multiply(hdulist[0].data[ap_number[i]+1],
                                       weight[i])
                    break
        R_full = np.sum(R, axis=0)
        cdelta1 = header0['CDELT1']
        crval1 = header0['CRVAL1']
        wav = np.linspace(0, len(R_full) - 1, len(R_full)) * cdelta1 + crval1

    return {'w': wav, 'R': R_full}


def rHARPS(FITSfile, obj=False, wavelenmin=False, date_obs=False,
           spec_bin=False, med_snr=False, hdnum=False, radvel=False,
           coeffs=False, SN=False):
    """Read a HARPS ADP FITS file and return a dictionary of information.
    Parameters
    ----------
    FITSfile : str or Path object
        A path to a HARPS FITS file to be read.
    obj : bool, Default: False
        If *True*, the output will contain the contents of the OBJECT FITS
        header card.
    wavelenmin : bool, Default: False
        If *True*, the output will contain the contents of the WAVELMIN FITS
        header card.
    date_obs : bool, Default: False
        If *True*, the output will contain the contents of the DATE-OBS FITS
        header card.
    spec_bin : bool, Default: False
        If *True*, the output will contain the contents of the SPEC_BIN FITS
        header card.
    med_snr : bool, Default: False
        If *True*, the output will contain the contents of the SNR FITS header
        card.
    hdnum : bool, Default: False
        If *True*, the output will contain the contents of the custom-added
        HDNUM FITS header card. (Added to unify object identifiers across all
        stars, some of which were occasionally identified by things other than
        HD number.)
    radvel : bool, Default: False
        If *True*, the output will contain the contents of the custom-added
        RADVEL FITS header card. (Added to unify the radial velocity for each
        star, as a small minority of stars had different radial velocity
        information in their HIERARCH ESO TEL TAFG RADVEL header cards.)
    coeffs : bool, Default: False
        If *True*, the output will contain the contents of the various
        *ESO DRS CAL TH COEFF LLX* header cards, where *X* ranges from 0 to
        287.
    Returns
    -------
    dict
        A dictionary containing the following key-value pairs:
        w : Numpy array
            The wavelength array.
        f : Numpy array
            The flux array.
        e : Numpy array
            The estimated error array (HARPS returns no error array by
            default).
        Optionally
        ==========
        obj : str
            The object name from the 'OBJECT' flag.
        wlmin : float
            The minimum wavelength.
        date_obs : datetime object
            The date the file was observed.
        spec_bin : float
            The wavelength bin size.
        med_snr : float
            The median SNR of the flux array.
        hd_num : str
            The HD identifier of the star in the format "HDxxxxxx".
        radvel : float
            The radial velocity of the star in km/s.
        If the `coeffs` keyword argument is *True*, there will be 288 entries
        of the form "ESO DRS CAL TH COEFF LLX": *value*, where X will range
        from 0 to 287.
    """

    result = {}
    try:
        with fits.open(FITSfile) as hdulist:
            try:
                header0 = hdulist[0].header
                header1 = hdulist[1].header
                data = hdulist[1].data
                w = data.WAVE[0]
                gain = header0['GAIN']
                # Multiply by the gain to convert from ADUs to photoelectrons
                f = data.FLUX[0] * gain
                e = 1.e6 * np.absolute(f)
                result['w'] = w
                result['f'] = f
                result['e'] = e
                if obj:
                    result['obj'] = header1['OBJECT']
                if wavelenmin:
                    result['wavelmin'] = header0['WAVELMIN']
                if date_obs:
                    result['date_obs'] = dt.datetime.strptime(
                            header0['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
                if spec_bin:
                    result['spec_bin'] = header0['SPEC_BIN']
                if med_snr:
                    result['med_snr'] = header0['SNR']
                if hdnum:
                    result['hdnum'] = header0['HDNUM']
                if radvel:
                    result['radvel'] = header0['RADVEL']
                if SN:
                    SNR = []
                    for i in range(72):
                        card = 'HIERARCH ESO DRS SPE EXT SN' + str(i)
                        SNR.append(header0[card])
                    result['SN'] = SNR

#               If the coeffs keyword is given, returna all
#               288 wavelength solution coefficients.
                if coeffs:
                    for i in range(0, 288, 1):
                        key_string = 'ESO DRS CAL TH COEFF LL{0}'.format(
                                str(i))
                        result[key_string] = header0[key_string]
                return result

            except:
                result['HAR'] = 1
                header0 = hdulist[0].header
                header1 = hdulist[1].header
                data = hdulist[1].data
                w = [1/x[0]*100000000 for x in np.flip(data)]
                # Multiply by the gain to convert from ADUs to photoelectrons
                f = [x[1] for x in np.flip(data)]
                result['w'] = w
                result['f'] = f
                result['e'] = np.divide(np.ones_like(w), 1000)
                if obj:
                    result['obj'] = header1['OBJECT']
                if wavelenmin:
                    result['wavelmin'] = header0['WAVELMIN']
                if date_obs:
                    result['date_obs'] = dt.datetime.strptime(
                            header0['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
                if spec_bin:
                    result['spec_bin'] = header0['SPEC_BIN']
                if med_snr:
                    result['med_snr'] = header0['SNR']
                if hdnum:
                    result['hdnum'] = header0['HDNUM']
                if radvel:
                    result['radvel'] = header0['RADVEL']
    #            if SN:
    #                SNR = []
    #                for i in range(72):
    #                    card = 'HIERARCH ESO DRS SPE EXT SN' + str(i)
    #                    SNR.append(header0[card])
    #                result['SN'] = SNR

        # If the coeffs keyword is given, returna all 288 wavelength solution
        # coefficients.
                if coeffs:
                    for i in range(0, 288, 1):
                        key_string = 'ESO DRS CAL TH COEFF LL{0}'.format(
                                str(i))
                        result[key_string] = header0[key_string]
                return result
    except OSError:
        with open(FITSfile) as ascii_table:
            w_line = ascii_table.readline()
            f_line = ascii_table.readline()
            w = [float(x) for x in w_line.split(',')]
            f = [float(x) for x in f_line.split(',')]
            result['w'] = w
            result['f'] = f
            result['e'] = np.absolute(np.divide(np.power(f, 0.4), 1000))
        return result


def rflatHARPS(FITSfile, obj=False, wavelenmin=False, date_obs=False,
               spec_bin=False, med_snr=False, hdnum=False, radvel=False,
               coeffs=False, SN=False):
    """Read a HARPS ADP FITS file and return a dictionary of information.
    Parameters
    ----------
    FITSfile : str or Path object
        A path to a HARPS FITS file to be read.
    obj : bool, Default: False
        If *True*, the output will contain the contents of the OBJECT FITS
        header card.
    wavelenmin : bool, Default: False
        If *True*, the output will contain the contents of the WAVELMIN FITS
        header card.
    date_obs : bool, Default: False
        If *True*, the output will contain the contents of the DATE-OBS FITS
        header card.
    spec_bin : bool, Default: False
        If *True*, the output will contain the contents of the SPEC_BIN FITS
        header card.
    med_snr : bool, Default: False
        If *True*, the output will contain the contents of the SNR FITS header
        card.
    hdnum : bool, Default: False
        If *True*, the output will contain the contents of the custom-added
        HDNUM FITS header card. (Added to unify object identifiers across all
        stars, some of which were occasionally identified by things other than
        HD number.)
    radvel : bool, Default: False
        If *True*, the output will contain the contents of the custom-added
        RADVEL FITS header card. (Added to unify the radial velocity for each
        star, as a small minority of stars had different radial velocity
        information in their HIERARCH ESO TEL TAFG RADVEL header cards.)
    coeffs : bool, Default: False
        If *True*, the output will contain the contents of the various
        *ESO DRS CAL TH COEFF LLX* header cards, where *X* ranges from 0 to
        287.
    Returns
    -------
    dict
        A dictionary containing the following key-value pairs:
        w : Numpy array
            The wavelength array.
        f : Numpy array
            The flux array.
        e : Numpy array
            The estimated error array (HARPS returns no error array by
            default).
        Optionally
        ==========
        obj : str
            The object name from the 'OBJECT' flag.
        wlmin : float
            The minimum wavelength.
        date_obs : datetime object
            The date the file was observed.
        spec_bin : float
            The wavelength bin size.
        med_snr : float
            The median SNR of the flux array.
        hd_num : str
            The HD identifier of the star in the format "HDxxxxxx".
        radvel : float
            The radial velocity of the star in km/s.
        If the `coeffs` keyword argument is *True*, there will be 288 entries
        of the form "ESO DRS CAL TH COEFF LLX": *value*, where X will range
        from 0 to 287.
    """

    result = {}
    with fits.open(FITSfile) as hdulist:
        header0 = hdulist[0].header
        f = hdulist[0].data
        cdelta1 = header0['CDELT1']
        crval1 = header0['CRVAL1']
        w = np.linspace(0, len(f), len(f)) * cdelta1 + crval1
        e = np.zeros(len(f))
        # Construct an error array by taking the square root of each flux value
        if SN:
            SNR = []
            for i in range(72):
                card = 'HIERARCH ESO DRS SPE EXT SN' + str(i)
                SNR.append(header0[card])
            result['SN'] = SNR

        result['w'] = w
        result['f'] = f
        result['e'] = e

    return result


def HAR2HER(spec, specres, pixelw, band_cor=True):

    if max(spec['w']) < 7885.0027:
        boun = [[4713.5737, 4901.3360], [5649.1206, 5872.0078],
                [6478.3989, 6736.1442]]
        npix = 4096  # number of pixels per band

        w = np.array(spec['w'])
        f = np.array(spec['f'])

        wreduced = [[], [], []]
        wspec = [[], [], [], []]
        freduced = [[], [], []]
        avpix = np.ones(3)
        sigma = np.ones(3)

        if band_cor is True:
            band_correction = [0.8, 0.775, 0.73]
        else:
            band_correction = [1., 1., 1.]

        for i in range(3):
            wreduced[i] = w[((boun[i][0] - 50) < w) & (w < (boun[i][1] + 50))]
            freduced[i] = f[((boun[i][0] - 50) < w) & (w < (boun[i][1] + 50))]
            aver = np.zeros(len(wreduced[i])-1)
            for j in range(len(wreduced[i])-1):
                aver[j] = wreduced[i][j+1] - wreduced[i][j]
                avpix[i] = np.average(aver)
            minus_pix = int(10.0/avpix[i])
            npixold = len(wreduced[i]) - minus_pix*2
            wspec[i] = np.linspace(boun[i][0]-40, boun[i][1]+40, num=npixold)
            freduced[i] = spectres(wspec[i], wreduced[i], freduced[i])
            wreduced[i] = wspec[i]
            avpix[i] = wreduced[i][1] - wreduced[i][0]
# Convolving the flux with gaussians (smooth them out)
# Calculate for each band the sigma (from HERMES) and Gaussian
        for j in range(3):
            sigma[j] = (boun[j][1] + boun[j][0]) / \
                        (2.0 * 2.355 * specres * band_correction[j] * avpix[j])
# For a broadened spectrum use the factor 2.25
            Gauss = Gaussian1DKernel(stddev=sigma[j])
# Convolve the flux with the Gaussian to "blur it out"
            freduced[j] = convolve(freduced[j], Gauss)

        wnew = [[], [], []]

        for j in range(3):
            wnew[j] = np.linspace(boun[j][0], boun[j][1], num=npix)
        enew = [np.zeros(npix), np.zeros(npix), np.zeros(npix)]

        fnew = [[], [], []]
        for i in range(3):
            try:
                fnew[i] = spectres(wnew[i], wreduced[i], freduced[i])
            except ValueError:
                return 0

        norm_flux = np.zeros_like(fnew)

        return {'w': wnew, 'f': fnew, 'e': enew, 'norm_f': norm_flux}

    else:
        boun = [[4713.5737, 4901.3360], [5649.1206, 5872.0078],
                [6478.3989, 6736.1442], [7585.0026, 7885.0027]]
        npix = 4096  # number of pixels per band

        w = np.array(spec['w'])
        f = np.array(spec['f'])

        wreduced = [[], [], [], []]
        wspec = [[], [], [], []]
        freduced = [[], [], [], []]

        avpix = np.ones(4)
        sigma = np.ones(4)

        if band_cor is True:
            band_correction = [0.8, 0.775, 0.75, 0.83]
        else:
            band_correction = [1., 1., 1., 1.]

        for i in range(4):
            wreduced[i] = w[((boun[i][0] - 50) < w) & (w < (boun[i][1] + 50))]
            freduced[i] = f[((boun[i][0] - 50) < w) & (w < (boun[i][1] + 50))]
            aver = np.zeros(len(wreduced[i])-1)
            for j in range(len(wreduced[i])-1):
                aver[j] = wreduced[i][j+1] - wreduced[i][j]
                avpix[i] = np.average(aver)
            minus_pix = int(10.0/avpix[i])
            npixold = len(wreduced[i]) - minus_pix*2
            wspec[i] = np.linspace(boun[i][0]-40, boun[i][1]+40, num=npixold)
            freduced[i] = spectres(wspec[i], wreduced[i], freduced[i])
            wreduced[i] = wspec[i]
            avpix[i] = wreduced[i][1] - wreduced[i][0]

# convolving the flux with gaussians (smooth them out)
# Calculate for each band the sigma (from HERMES) and Gaussian

        for j in range(4):
            sigma[j] = (boun[j][1] + boun[j][0]) / \
                        (2.0 * 2.355 * specres * band_correction[j] * avpix[j])
# For a broadened spectrum use the factor 2.25
            Gauss = Gaussian1DKernel(stddev=sigma[j])
# convolve the flux with the Gaussian to "blur it out"
            freduced[j] = convolve(freduced[j], Gauss)

        wnew = [[], [], [], []]

        for j in range(4):
            wnew[j] = np.linspace(boun[j][0], boun[j][1], num=npix)
        enew = [np.zeros(npix), np.zeros(npix), np.zeros(npix), np.zeros(npix)]

        fnew = [[], [], [], []]

        for i in range(4):
            try:
                fnew[i] = spectres(wnew[i], wreduced[i], freduced[i])
            except ValueError:
                return 0

        norm_flux = np.zeros_like(fnew)

        return {'w': wnew, 'f': fnew, 'e': enew, 'norm_f': norm_flux}


def HAR2HER2(spec):
    boun = [[4713.5737, 4901.3360], [5649.1206, 5872.0078],
            [6478.3989, 6736.1442], [7585.0026, 7885.0027]]
    buf = [[5.5, 5.7, 11.2], [6.6, 6.9, 13.5], [7.6, 7.9, 15.5],
           [8.9, 9.2, 18.1]]

    w = np.array(spec['w'])
    f = np.array(spec['f'])

    wreduced = [[], [], [], []]
    freduced = [[], [], [], []]
    wnew = [[], [], [], []]
    fnew = [[], [], [], []]
    enew = [[], [], [], []]

    if spec['w'][-1] < 7500:
        bands = 3
    else:
        bands = 4

    for i in range(bands):
        wreduced[i] = w[((boun[i][0] - buf[i][0] - 1) < w) &
                        (w < (boun[i][1] + buf[i][1] + 1))]
        freduced[i] = f[((boun[i][0] - buf[i][0] - 1) < w) &
                        (w < (boun[i][1] + buf[i][1] + 1))]

        pix_diff = np.array([])
        for j in range(len(wreduced)-1):
            pix_diff = np.append(pix_diff, wreduced[i][j+1] - wreduced[i][j])
            step_size = np.max(pix_diff)
            npix = int((boun[i][1]+buf[i][2] - boun[i][0]) / step_size)

        wnew[i] = np.linspace(boun[i][0]-buf[i][0], boun[i][1]+buf[i][1],
                              num=npix)
        fnew[i] = spectres(wnew[i], wreduced[i], freduced[i])

        enew[i] = np.zeros_like(fnew[i])

    norm_flux = np.zeros_like(fnew)

    return {'w': wnew, 'f': fnew, 'e': enew, 'norm_f': norm_flux,
            'norm_f2': norm_flux, 'e2': enew}


def rHERMES_prep(FITSfile, datahdu=0, SN=False):
    """Read a HERMES FITS file and returns data information.
    Parameters
    ----------
    FITSfile : str or Path object
        A path to a HARPS FITS file to be read.
    datahdu  : int
        Decides which data hdulist to read the data from
        0 is Flux, 4 is normalized Flux in HERMES spectra.

    Returns
    -------
    dict
        A dictionary containing the following key-value pairs:
        w : Numpy array
            The wavelength array.
        f : Numpy array
            The flux array.
        e : Numpy array
            A zero array to be changed later
    """

    result = {}
    if FITSfile.endswith('.fits'):
        with fits.open(FITSfile) as hdulist:
            header0 = hdulist[0].header
            f = hdulist[datahdu].data
            sky_f = hdulist[2].data
            try:
                norm_f = hdulist[4].data
            except IndexError:
                norm_f = np.ones_like(f)
            e = hdulist[1].data
            cdelta1 = header0['CDELT1']
            crval1 = header0['CRVAL1']
#            create wavelength and error (only 0 values by this point) array
            w = np.linspace(0, len(f) - 1, len(f)) * cdelta1 + crval1
#            If we want to use the normalized spectrum, we should use a
#            normalized error
#            if datahdu == 4:
#                e = np.divide(np.multiply(e, f), hdulist[0].data)
#            write array on output
            result['w'] = w
            result['f'] = f
            result['sky_f'] = sky_f
            result['e'] = e
            result['norm_f'] = norm_f
#            print(e)
            result['disp'] = w[1] - w[0]

            if SN and 'SNR' in header0:
                SNR = header0['SNR']
                result['SNR'] = SNR
            else:
                result['SNR'] = 1000

        return result

    else:
        wave = np.array([], dtype='float64')
        flux = np.array([], dtype='float64')

        with open(FITSfile) as data:
            for line in data:
                if line.startswith('#'):
                    continue
                wave = np.append(wave, float(line.split(',')[0]))
                flux = np.abs(np.append(flux, float(line.split(',')[1])))

        result['w'] = wave
        result['f'] = flux
        result['e'] = np.absolute(np.divide(np.power(flux, 0.4), 1000))
        result['SNR'] = 1000

        return result


def get_spec_wave(fname):
    hdulist = fits.open(fname)
    hdu = hdulist[0]
    crpix = hdu.header['CRPIX1']-1.0
    crval = hdu.header['CRVAL1']
    cdelt = hdu.header['CDELT1']

    spec_wave = np.float32(np.arange(4096))
    for i in np.arange(4096):
        spec_wave[i] = (spec_wave[i]-crpix)*cdelt+crval
    print(spec_wave)
    return spec_wave


def read_unreduced(unred_spec):
    fitsspec = {}
    spec_wave = get_spec_wave(unred_spec)
    hdulist = fits.open(unred_spec)
    flux = hdulist[0].data
    err = np.sqrt(np.abs(hdulist[1].data))
    sky_f = hdulist[5].data
    name = hdulist[2].data['NAME']
    err = np.array(err)
#    err that is nan or smaller than 0 is unacceptable,
#    so they are set to the max error
    for j in range(len(err)):
        err[j][err[j] != err[j]] = np.ones_like(err[j][err[j] != err[j]]) * \
            np.max(err[j][err[j] == err[j]])
        err[j][err[j] <= 0] = np.ones_like(err[j][err[j] <= 0]) * \
            np.max(err[j])

    fitsspec['w'] = spec_wave
    fitsspec['f'] = flux
    fitsspec['e'] = err
    fitsspec['sky_f'] = sky_f
    fitsspec['norm_sky_f'] = np.zeros_like(sky_f)
    fitsspec['disp'] = spec_wave[1] - spec_wave[0]
    fitsspec['name'] = name
    fitsspec['norm_f'] = np.zeros_like(fitsspec['f'])
    fitsspec['band'] = int(spec_wave[0] / 1000) - 3
    return fitsspec


def make_cuts(array, n_parts):
    """
    Cuts a given `array` up into `n_parts` equal parts, while making sure that
    each part contains half of the previous and half of the next part (unless
    it is the first or last part).

    Parameters
    ----------
    array : array_like
        The array that needs to be cut into `n_parts`.
    n_parts : int
        The number of parts that `array` needs to be cut into.

    Returns
    -------
    parts : list of :obj:`~numpy.ndarray` objects
        List containing the created parts.

    Example
    -------
    Dividing a sequence of six integers up into 5 parts:

        >>> array = [0, 1, 2, 3, 4, 5]
        >>> make_cuts(array, 5)
        [array([0., 1.]),
         array([1., 2.]),
         array([2., 3.]),
         array([3., 4.]),
         array([4., 5.])]

    """

    # Make sure that array is a NumPy array
    array = np.asarray(array)

    # Calculate the number of elements in array
    len_tot = np.shape(array)[0]

    # Determine number of cuts
    n_cuts = int(np.ceil(n_parts/2))

    # Determine the stepsize of all boundaries
    step_size = len_tot/n_cuts/2

    # Calculate the lengths of all bins
    bins = np.array(np.ceil(np.arange(0, len_tot+1, step_size)), dtype=int)

    # Divide array up into n_parts parts, using two sequential bins each time
    parts = [array[i:j] for i, j in zip(bins[:-2], bins[2:])]

    # Return parts
    return(parts)


def norm_flux(w, f, res_w, e=[], rej=0.75, l_order=3, Kurucz=False):
    f1, f2, f3, f4, w1, w2, w3, index1, index2 = [], [], [], [], [], [], [],\
                                                 [], []
#    plt.step(w, f)
#    plt.axhline(0)
#    plt.show()
    iterations = 0
    if all(ws > 4500 for ws in w) and all(ws < 5000 for ws in w):
        sd_correct = 3
    else:
        sd_correct = 1

    delete_con = np.argwhere([4850.0 <= ws <= 4875.0 or 6549.0 <= ws <=
                              6581.0 or 7587 <= ws <= 7689 for ws in w])
    if Kurucz is True:
        delete_con = np.argwhere([4855.0 <= ws <= 4870.0 or 6550.0 <= ws <=
                                  6575.0 or 7587 <= ws <= 7689 for ws in w])

    w1 = np.delete(w, delete_con)
    f1 = np.delete(f, delete_con)

    delete_con2 = np.argwhere([fs <= 0.1 for fs in f1])
#    print(len(delete_con2), len(f1))
    if len(f1)*0.1 <= len(delete_con2):
        print('Warning: Flux values of a part of the spectrum is below 0')
        return [-1], [-1]

    w1 = np.delete(w1, delete_con2)
    f1 = np.delete(f1, delete_con2)

    med_f = np.median(f1)
    if len(e) != 0:
        e1 = np.delete(e, delete_con)
        e1 = np.delete(e1, delete_con2)
        e1 = np.delete(e1, np.argwhere([fs > 2.5*med_f for fs in f1]))
        med_e = np.median(e1)
        e1 = np.where([e < med_e/3 for e in e1], med_e*1000, e1)
        e3 = e1

    w1 = np.delete(w1, np.argwhere([fs > 2.5*med_f for fs in f1]))
    f1 = np.delete(f1, np.argwhere([fs > 2.5*med_f for fs in f1]))

    f3 = np.array(f1)
    w3 = np.array(w1)

    w11 = w1[:int(len(w1)/2)]
    w12 = w1[int(len(w1)/2):]
    f11 = f1[:int(len(w1)/2)]
    f12 = f1[int(len(w1)/2):]
    if len(e) != 0:
        e11 = e1[:int(len(w1)/2)]
        e12 = e1[int(len(w1)/2):]

    for i in range(int(rej*len(f11))):
        index1.append(np.argmin(f11))
        f11[index1[i]] = 1000000000000
    for i in range(int(rej*len(f12))):
        index2.append(np.argmin(f12))
        f12[index2[i]] = 1000000000000

    w2 = np.concatenate([np.delete(w11, index1), np.delete(w12, index2)])
    f2 = np.concatenate([np.delete(f11, index1), np.delete(f12, index2)])

    if len(e) != 0:
        e2 = np.concatenate([np.delete(e11, index1), np.delete(e12, index2)])
        e22 = np.square(e2)
        if len(e2[e22 == 0]) == 0:
            weight = 1 / np.square(e2)
            if np.any(np.isinf(weight)):
                weight = np.ones_like(w2) / np.std(f2)
        else:
            weight = np.ones_like(w2) / np.std(f2)
    else:
        weight = np.ones_like(w2) / np.std(f2)
    med_weig = np.median(weight)
    for ind in range(len(weight)):
        if weight[ind] > 2.5 * med_weig:
            weight[ind] = 0

    coef = np.polynomial.legendre.legfit(w2, f2, l_order, w=weight)
    fit = np.polynomial.legendre.legval(w3, coef)
    sdev = np.std(np.subtract(f3, fit))

    while True:
        up = 3.0 * sdev
        down = -1.5 * sdev
        dev = np.subtract(f3, fit)
        f4 = np.delete(f3,
                       np.argwhere([devs > up or devs < down for devs in dev]))
        w4 = np.delete(w3,
                       np.argwhere([devs > up or devs < down for devs in dev]))

        if iterations > 100:
            print("Warning: Reached 100 iterations on the continuum fit.")
            break
        if np.array_equal(f3, f4):
            break
        if len(e) == 0:
            weight = np.ones_like(w4) / sdev
        else:
            e4 = np.delete(
                 e3, np.argwhere([devs > up or devs < down for devs in dev]))
            weight = 1 / np.square(e4)
            if np.any(np.isinf(weight)):
                weight = np.ones_like(w4) / sdev
        if len(w4) == 0:
            print('Error: Fit could not stabilise!')
            return [-1], [-1]

        coef = np.polynomial.legendre.legfit(w4, f4, l_order, w=weight)
        fit = np.polynomial.legendre.legval(w4, coef)
        sdev = np.std(np.subtract(f4, fit))

        f3 = f4
        w3 = w4
        if len(e) != 0:
            e3 = e4

        iterations = iterations + 1

    fit = np.polynomial.legendre.legval(res_w, coef)
    fit_e_sdev = sdev / (fit * sd_correct)

#    plt.plot(w, f)
#    plt.plot(res_w, fit)
#    plt.plot(w2, f2)
#    plt.show()
#    plt.clf()

    return fit, fit_e_sdev


def splice_spectra(wav, flux):

    """
    Cuts a given `array` up into `n_parts` equal parts, while making sure that
    each part contains half of the previous and half of the next part (unless
    it is the first or last part).

    Parameters
    ----------
    wav : array_like
        The array containing the cut wavelength arrays
    flux : array_like
        The array containing the cut flux arrays (normalized).

    Returns
    -------
    array : np_array
        The array containing the complete spectrum.

    """

#    find the index where each array overlaps with a new one
    splind = []
    w = []
    f = []
    for i in range(len(wav)-1):
        splitpoint = wav[i+1][0]
        splind = np.concatenate((splind,
                                 np.argwhere(abs(wav[i] - splitpoint)
                                             < 0.001)[0]))

    splitpoint = wav[-2][-1]
    splind = np.concatenate((splind, np.argwhere(abs(wav[-1] - splitpoint)
                                                 < 0.001)[0] + 1))
    splind = splind.astype(int)
#        Get the first part of the spectrum (no overlap)
    for i in range(len(wav)+1):
        if i == 0:
            f = np.concatenate((f, np.split(flux[i], [0, splind[i]])[1]))
            w = np.concatenate((w, np.split(wav[i], [0, splind[i]])[1]))
#        Get the middle part of the spectrum by weigthing the overlapping parts
        elif i < len(wav):
            weight = np.linspace(0, 1, len(np.split(flux[i],
                                                    [0, splind[i]])[1]))
            f = np.concatenate((f, np.split(flux[i], [0, splind[i]])[1] *
                                weight + np.split(flux[i-1],
                                                  [0, splind[i-1]])[2] *
                                (1 - weight)))
            w = np.concatenate((w, np.split(wav[i], [0, splind[i]])[1]))
#        Get the last part of the spectrum (no overlap)
        elif i == len(wav):
            f = np.concatenate((f, np.split(flux[i-1],
                                            [0, splind[i-1]])[2]))
            w = np.concatenate((w, np.split(wav[i-1],
                                            [0, splind[i-1]])[2]))

    return w, f


def normalize_HERMES(w, f, w_array, e=[], title='Unknown', Kurucz=False,
                     plot_cuts=False):
    """
    Normalizes a simulated spectrum in HERMES resolution and returns the
    continuum function for the wavelength array w_array.

    Parameters
    ----------
    norm_spec : spectrum
        A library containing the flux, wavelength and error (probably empty)
        of a simulated spectrum that will be used for the normalization
        process.
    w_array : array_like
        Wavelength array that is used to define the resulting continuum
        function on.

    Returns
    -------
    array : np_array
        The array containing the complete spectrum.

    """
    if len(f) != len(w):
        print('Error: Length of flux and wavelength arrays are different')
        return [-1], [-1]
    fcon = [f == f]
    continuum_func = np.ones_like(f)
    if len(e) != 0:
        if len(f) != len(e):
            print('Error: Length of flux and error arrays are different')
            return [-1], [-1]
        e = e[tuple(fcon)]
    w_array = np.array(w_array)
    w_array = w_array[tuple(fcon)]
    w = w[tuple(fcon)]
    f = f[tuple(fcon)]

    if any(4700 <= ws <= 5900 for ws in w):
        n = 6
    elif any(7500 <= ws for ws in w):
        n = 4
    else:
        n = 6

#        Split the wavelength and flux array into equal parts
    cutw = make_cuts(w, n)
    cutf = make_cuts(f, n)

    con = np.bitwise_and(w[0] < w_array, w_array < w[-1])
    continuum_cutw = make_cuts(w_array[con], n)

    attach1 = np.argwhere(w_array == continuum_cutw[0][0])[0][0]
    attach2 = np.argwhere(w_array == continuum_cutw[-1][-1])[0][0]

    continuum_cutw[0] = np.concatenate((w_array[:attach1], continuum_cutw[0]))
    continuum_cutw[-1] = np.concatenate((continuum_cutw[-1],
                                         w_array[attach2+1:]))

    continuum_cutf = np.zeros_like(continuum_cutw)
    continuum_e = np.zeros_like(continuum_cutw)

    if len(e) != 0:
        cute = make_cuts(e, n)
    else:
        cute = [[], [], [], [], [], [], [], [], [], []]

    for j in range(len(cutf)):
        if j == 0:
            continuum_cutf[j], continuum_e[j] = norm_flux(cutw[j], cutf[j],
                                                          continuum_cutw[j],
                                                          e=cute[j],
                                                          Kurucz=Kurucz)
        elif j < n-2 and j > 0:
            continuum_cutf[j], continuum_e[j] = norm_flux(cutw[j], cutf[j],
                                                          continuum_cutw[j],
                                                          e=cute[j],
                                                          Kurucz=Kurucz)
        elif j == n-2:
            continuum_cutf[j], continuum_e[j] = norm_flux(cutw[j], cutf[j],
                                                          continuum_cutw[j],
                                                          e=cute[j],
                                                          Kurucz=Kurucz)
        if continuum_cutf[j][0] == -1:
            return [-1], [-1]

    w_array2, continuum_func[tuple(fcon)] = splice_spectra(continuum_cutw,
                                                           continuum_cutf)
    w_array3, err_continuum = splice_spectra(continuum_cutw, continuum_e)

    if plot_cuts is True:
        pdf = matplotlib.backends.backend_pdf.PdfPages('plot_cut.pdf')
        fig, ax = plt.subplots(nrows=1, ncols=1)
        factor = 1.15
#        color1 = 'red'
        ax.step(w, f, label='HERMES spectrum')
        ax.step(w_array, continuum_func[tuple(fcon)], '--', lw=5,
                label='Continuum fit')
        ax.set_ylim(min(f)*0.8, max(f)*1.2)
        ax.set_xlim(min(w)-10, max(w)+10)
        ax.axvline(continuum_cutw[0][0], color='black')
        ax.axvline(continuum_cutw[1][0], color='black')
        ax.set_xlabel(r'Wavelength [\AA]')
        ax.set_ylabel('Counts [ADU]')
        ax.yaxis.set_major_locator(MultipleLocator(500))
        for cont in continuum_cutw:
            ax.axvline(cont[-1], color='black')
#            ax.annotate(text='', xy=(cont[0], max(f)*factor),
#                        xytext=(cont[-1], max(f)*factor),
#                        arrowprops=dict(arrowstyle='<->, head_width=0.75',
#                                        lw=3))
            if factor > 1.1:
                factor = factor - 0.1
            else:
                factor = factor + 0.1
        ax.legend(loc='lower left', handlelength=1.0)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98,
                            wspace=0.0, hspace=0.0)
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0.02)
        pdf.close()
        plt.clf()

    return continuum_func, err_continuum


def wHERMES(Specdat, inputname, outputname,
            HERheader='/home/christian/.config/HER_example/' +
            '1702160018012971.fits',
            SNR=[0, 0, 0], SNRpp=[0, 0, 0], bands=4):
    outputname = outputname.rsplit(".", 1)[0]
    for i in range(bands):
        with fits.open(HERheader) as hdulist:
            header0 = hdulist[0].header
            header4 = hdulist[4].header
        with fits.open(inputname) as hdulist:
            header1 = hdulist[0].header

# Wavelenght array information
        header0['CRPIX1'] = 1.0
        header0['CRVAL1'] = Specdat['w'][i][0]
        header0['CDELT1'] = Specdat['w'][i][1]-Specdat['w'][i][0]
        header0['NAXIS1'] = len(Specdat['w'][i])

        header4['CRPIX1'] = 1.0
        header4['CRVAL1'] = Specdat['w'][i][0]
        header4['CDELT1'] = Specdat['w'][i][1]-Specdat['w'][i][0]
        header4['NAXIS1'] = len(Specdat['w'][i])

# copy header information from input
        if 'DATE-OBS' in header1:
            header0['DATE-OBS'] = header1['DATE-OBS']
        if 'OBJECT' in header1:
            header0['OBJECT'] = header1['OBJECT']
        if 'EXPOSED' in Specdat:
            header0['EXPOSED'] = Specdat['EXPOSED']
            header0['TOTALEXP'] = Specdat['EXPOSED']
        header0['SNR'] = SNR[i]
        header0['SNRpp'] = SNRpp[i]

# put some of my own information into header
        header0['ORIGIN'] = 'MY OWN SICK MIND'
        header0['DATE'] = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        header0['IRAF-TLM'] = dt.datetime.now().strftime("%Y-%m-\
               %dT%H:%M:%S")

        PRIMARY = fits.PrimaryHDU(data=Specdat['f'][i], header=header0)
        EXT1 = fits.ImageHDU(data=(Specdat['e'][i]), header=header0,
                             name='input_sigma')
        if 'norm_sky_f' in Specdat:
            EXT2 = fits.ImageHDU(data=Specdat['norm_sky_f'][i], header=header0,
                                 name='no_sky_subspectrum')
        else:
            EXT2 = fits.PrimaryHDU(data=Specdat['f'][i], header=header0,
                                   name='no_sky_subspectrum')
        EXT3 = fits.ImageHDU(data=np.zeros(len(Specdat['f'][i])),
                             header=header0, name='no_sky_sigma')
        EXT4 = fits.ImageHDU(data=(Specdat['norm_f'][i]),
                             header=header4)
        hdul = fits.HDUList([PRIMARY, EXT1, EXT2, EXT3, EXT4])

        hdul.writeto(outputname + str(i+1) + '.fits', overwrite=True)


def wHERMES2(Specdat, inputname, outputname,
             HERheader='/home/christian/.config/HER_example/' +
             '1702160018012971.fits'):
    try:
        with fits.open(HERheader) as hdulist:
            header0 = hdulist[0].header
            header4 = hdulist[4].header
    except FileNotFoundError:
        with fits.open(inputname) as hdulist:
            header0 = hdulist[0].header
            header4 = hdulist[4].header
    with fits.open(inputname) as hdulist:
        header1 = hdulist[0].header

# Wavelenght array information
    header0['CRPIX1'] = -1.0
    header0['CRVAL1'] = Specdat['w'][0]
    header0['CDELT1'] = (Specdat['w'][-1]-Specdat['w'][0]) / 4095
    header0['NAXIS1'] = len(Specdat['w'])

    header4['CRPIX1'] = 1.0
    header4['CRVAL1'] = Specdat['w'][0]
    header4['CDELT1'] = Specdat['w'][1]-Specdat['w'][0]
    header4['NAXIS1'] = len(Specdat['w'])

# copy header information from input
    if 'DATE-OBS' in header1:
        header0['DATE-OBS'] = header1['DATE-OBS']
    if 'OBJECT' in header1:
        header0['OBJECT'] = header1['OBJECT']
    if 'EXPOSED' in Specdat:
        header0['EXPOSED'] = Specdat['EXPOSED']
        header0['TOTALEXP'] = Specdat['EXPOSED']


# put some of my own information into header
    PRIMARY = fits.PrimaryHDU(data=Specdat['f'], header=header0)
    EXT1 = fits.ImageHDU(data=(Specdat['e']), header=header0,
                         name='input_sigma')
    if 'norm_sky_f' in Specdat:
        EXT2 = fits.ImageHDU(data=Specdat['norm_sky_f'],
                             header=header0, name='no_sky_subspectrum')
    else:
        EXT2 = fits.ImageHDU(data=Specdat['f'],
                             header=header0, name='no_sky_subspectrum')
    EXT3 = fits.ImageHDU(data=np.zeros(len(Specdat['f'])),
                         header=header0, name='no_sky_sigma')
    EXT4 = fits.ImageHDU(data=(Specdat['norm_f']),
                         header=header4, name='HERMES_normalization')
    hdul = fits.HDUList([PRIMARY, EXT1, EXT2, EXT3, EXT4])

    hdul.writeto(outputname, overwrite=True)
