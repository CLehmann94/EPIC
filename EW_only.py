#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 15:45:33 2021

@author: christian
"""

import gzip
import math
import numpy as np
from scipy.optimize import curve_fit
from EPIC_functions import dTemp_lin, dlogg_lin, dMetal


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


def EW_only(EW_file, lineparams, casali_corr, Teff_corr, logg_corr, feh_corr,
            par=True):

    with gzip.open(EW_file, mode="rt") as EW_dat:
        lines = EW_dat.readlines()
        suc_ele = np.array([], dtype='str')
        suc_ion = np.array([], dtype='float64')
        suc_line = np.array([], dtype='float64')
        EWRef = np.array([], dtype='float64')
        EWR_sig = np.array([], dtype='float64')
        EWTar = np.array([], dtype='float64')
        EWT_sig = np.array([], dtype='float64')

        for i, line in enumerate(lines):
            if line.startswith(';') or line.startswith('#'):
                continue
            if len(line.split()) == 10:
                continue
            suc_ele = np.append(suc_ele, line.split()[0])
            suc_ion = np.append(suc_ion, float(line.split()[1]))
            suc_line = np.append(suc_line, float(line.split()[2]))
            EWRef = np.append(EWRef, float(line.split()[3]))
            EWR_sig = np.append(EWR_sig, float(line.split()[4]))
            EWTar = np.append(EWTar, float(line.split()[5]))
            EWT_sig = np.append(EWT_sig, float(line.split()[6]))


#    Calculating the Stellar Parameters using the stacked HERMES spectral data.
    if par is True:
        EW2 = np.array([], dtype='float64')
        EW2_sig = np.array([], dtype='float64')
        ele2 = np.array([], dtype='float64')
        io2 = np.array([], dtype='int')
        line2 = np.array([], dtype='float64')
        a1 = np.array([], dtype='float64')
        a1err = np.array([], dtype='float64')
        a2 = np.array([], dtype='float64')
        a2err = np.array([], dtype='float64')
        b1 = np.array([], dtype='float64')
        b1err = np.array([], dtype='float64')
        b2 = np.array([], dtype='float64')
        b2err = np.array([], dtype='float64')
        d1 = np.array([], dtype='float64')
        d1err = np.array([], dtype='float64')
        d2 = np.array([], dtype='float64')
        d2err = np.array([], dtype='float64')
        com1 = np.array([], dtype='float64')
        com1err = np.array([], dtype='float64')
        off = np.array([], dtype='float64')
        offerr = np.array([], dtype='float64')
        indic = np.array([], dtype='int')
        sl = 0
        snl = 0

        with open(lineparams) as lpar:
            for parline in lpar:
                if parline.startswith('#'):
                    continue
                lin = [i for i in parline.split(' ') if i]
                ele2 = np.append(ele2, str(lin[0]))
                io2 = np.append(io2, int(lin[1]))
                line2 = np.append(line2, float(lin[2]))
                a1 = np.append(a1, float(lin[3]))
                a1err = np.append(a1err, float(lin[4]))
                a2 = np.append(a2, float(lin[5]))
                a2err = np.append(a2err, float(lin[6]))
                b1 = np.append(b1, float(lin[7]))
                b1err = np.append(b1err, float(lin[8]))
                b2 = np.append(b2, float(lin[9]))
                b2err = np.append(b2err, float(lin[10]))
                d1 = np.append(d1, float(lin[11]))
                d1err = np.append(d1err, float(lin[12]))
                d2 = np.append(d2, float(lin[13]))
                d2err = np.append(d2err, float(lin[14]))
                com1 = np.append(com1, float(lin[15]))
                com1err = np.append(com1err, float(lin[16]))
                off = np.append(off, float(lin[17]))
                offerr = np.append(offerr, float(lin[18]))
        for i in range(len(ele2)):
            p1 = [x == ele2[i] for x in suc_ele]
            p2 = [x == io2[i] for x in suc_ion]
            p3 = [math.floor(x*10)/10 == line2[i] for x in suc_line]
            pos = np.argwhere(np.bitwise_and(np.bitwise_and(p1, p2), p3))
            if len(pos) == 1:
                EW2 = np.append(EW2, EWRef[pos[0][0]]-EWTar[pos[0][0]])
                EW2_sig = np.append(EW2_sig, np.sqrt(
                        np.square(EWT_sig[pos[0][0]]) +
                        np.square(EWR_sig[pos[0][0]])))
                indic = np.append(indic, i)
                snl += 1
            else:
                sl += 1
        print(len(EW2))

        initialParameters = [5750.0, 4.4, 0.1]
        parabound = ([5000, 3.2, -0.8], [6500, 5.0, 0.55])

        try:
            EW_fit_sig = np.sqrt(np.square(EW2_sig) +
                                 hypererr(initialParameters, [a1err[indic],
                                                              a2err[indic],
                                                              b1err[indic],
                                                              b2err[indic],
                                                              d1err[indic],
                                                              d2err[indic],
                                                              com1err[indic],
                                                              offerr[indic]]))
            popt, pcov = curve_fit(hypersurfacelstsq, [a1[indic], a2[indic],
                                                       b1[indic], b2[indic],
                                                       d1[indic], d2[indic],
                                                       com1[indic],
                                                       off[indic]], EW2,
                                   p0=initialParameters, sigma=EW_fit_sig,
                                   bounds=parabound)

            lstsq = popt
            lstsq_sig = np.sqrt(np.diag(pcov))

        except RuntimeError:
            EW_fit_sig = np.sqrt(np.square(EW2_sig) + hypererr(
                    initialParameters, [a1err[indic], a2err[indic],
                                        b1err[indic], b2err[indic],
                                        d1err[indic], d2err[indic],
                                        com1err[indic], offerr[indic]]))
            popt, pcov = curve_fit(hypersurfacelstsq,
                                   [a1[indic], a2[indic], b1[indic], b2[indic],
                                    d1[indic], d2[indic], com1[indic],
                                    off[indic]], EW2, p0=initialParameters,
                                   sigma=EW_fit_sig)

            lstsq = popt
            lstsq_sig = np.sqrt(np.diag(pcov))
        except RuntimeError:
            lstsq = [0, 0, 0]
            lstsq_sig = [0, 0, 0]

        if casali_corr is True:
            T_cas = dTemp_lin(lstsq, Teff_corr[0], Teff_corr[1], Teff_corr[2],
                              Teff_corr[3], Teff_corr[4])
            logg_cas = dlogg_lin(lstsq, logg_corr[0], logg_corr[1],
                                 logg_corr[2], logg_corr[3], logg_corr[4])
            feh_cas = dMetal(lstsq, feh_corr[0], feh_corr[1], feh_corr[2],
                             feh_corr[3], feh_corr[4])
            lstsq[0] = lstsq[0] - T_cas
            lstsq[1] = lstsq[1] - logg_cas
            lstsq[2] = lstsq[2] - feh_cas

        if EW_file.split('/')[-1].startswith('Norm_'):
            out_name = EW_file.split('/')[-1].replace('Norm_', '')
        elif EW_file.split('/')[-1].startswith('HERMES_'):
            out_name = EW_file.split('/')[-1].replace('HERMES_', '')
        else:
            out_name = EW_file.split('/')[-1]
        if out_name.endswith('EW.dat'):
            out_name = out_name.replace('EW.dat', '')

        with open('stellar_params_EW_only.dat', 'a') as out:
            out.write(('{:>5s} {:>11.1f} {:>11.1f} {:>11.3f}'
                      + '{:>11.3f} {:>11.4f} {:>11.4f}\n').format(
                      out_name, lstsq[0],
                      lstsq_sig[0], lstsq[1], lstsq_sig[1], lstsq[2],
                      lstsq_sig[2]))
#        if 5672 < lstsq[0] < 5872 and 4.24 < lstsq[1] < 4.64 and \
#           -0.1 < lstsq[2] < 0.1:
#            with open('SP1_EW_only.dat', 'a') as out:
#                out.write(('{:>5s} {:>11.1f} {:>11.1f} {:>11.3f} '
#                           + '{:>11.3f} {:>11.4f} {:>11.4f}\n').format(
#                            out_name, lstsq[0], lstsq_sig[0],
#                            lstsq[1], lstsq_sig[1], lstsq[2], lstsq_sig[2]))
#
#        if 5572 < lstsq[0] < 5972 and 4.14 < lstsq[1] < 4.74 and \
#           -0.2 < lstsq[2] < 0.2:
#            with open('SP2_EW_only.dat', 'a') as out:
#                out.write(('{:>5s} {:>11.1f} {:>11.1f} {:>11.3f}'
#                           + '{:>11.4f} {:>11.4f}\n').format(
#                            out_name, lstsq[0], lstsq_sig[0],
#                            lstsq[1], lstsq_sig[1], lstsq[2], lstsq_sig[2]))
#
#        if 5472 < lstsq[0] < 6072 and 4.04 < lstsq[1] < 4.84 and \
#           -0.3 < lstsq[2] < 0.3:
#            with open('SP3_EW_only.dat', 'a') as out:
#                out.write(('{:>5s} {:>11.1f} {:>11.1f} {:>11.3f}'
#                           + '{:>11.3f} {:>11.4f} {:>11.4f}\n').format(
#                            out_name, lstsq[0], lstsq_sig[0],
#                            lstsq[1], lstsq_sig[1], lstsq[2], lstsq_sig[2]))
