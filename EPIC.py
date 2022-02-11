#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 18:12:52 2020

@author: christian
"""

# Python modules
from astropy import constants as const
from astropy.io import fits
import gzip
import math
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os
from pathlib import Path
from scipy.optimize import curve_fit
from spectres import spectres
import sys
import unyt as u

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# My modules
from EPIC_functions import Gauss, parabel, hypersurface, hypersurfacelstsq, \
                           hypererr, gauss_function, linear_SNR, read_sky, \
                           find_nearest_idx, pivot_to_ap, renorm, dTemp_lin, \
                           dlogg_lin, dMetal
from EPIC_scripts import addSN, addSN_simple, add_weight, air2vacESO, \
                      center_line, determine_radvel, prepare_reference_rv, \
                      prepare_reference, lineup, line_prep_plot, measure_EW, \
                      prepare_target_wavelegnth, readlinelistw, \
                      read_ap_correct, combine_ap, rHERMES, r_resolving_map, \
                      determine_resolving_power
from EW_only import EW_only


# Help Message
def helpmessage(PROGNAME):
    print('')
    print('A program to determine equivalent widths of target spectra'
          ' compared to a reference spectrum. These differences get converted'
          ' into stellar parameters and saved into a stellar_params.dat file.'
          ' The raw EW data can be found in *target_name*EW.dat')
    print('')
    print('Usage: ', PROGNAME, '(Options) [1.reference / 2.target'
          ' HERMES witout n.fits]')
    print('')
    print('Options:  --help           Print this message.')
    print('          -e, --err t V    Add noise to the target spectrum'
          ' by giving')
    print('                           exposure time(s), mag of V band'
          ' and base noise.')
    print('          -h, --harps      When using simulated HERMES spectra '
          '(from HARPS).')
    print('          -s, --self       When using the Kurucz atlas for both '
          'reference and')
    print('                           target spectrum')
    print('          --EW_only        Read in the EW.dat file instead of a')
    print('                           spectrum (last argument is EW file)')
    print('')
    sys.exit(3)


def main_EPIC(argv=[], spec_name='', ref_name='', reduce_out=False):
    if len(argv) == 0:
        argv = sys.argv
        k = 0
    else:
        k = 1
    if len(spec_name) == 0 or len(ref_name) == 0:
        spec_name = argv[-1]
        ref_name = argv[-2]
    PROGNAME = argv[0].split('/')[-1]
    argc = len(argv)
    callibration = os.path.dirname(os.path.realpath(__file__)) + \
        '/calibration/'


# looking for help?
    if argc + k <= 2:
        helpmessage(PROGNAME)
    else:
        for i in range(argc):
            if argv[i] == '--help':
                helpmessage(PROGNAME)


#   numbers setup
    ap_number = [0]
    ap_weight = [1]
    c = const.c.to('km/s')
    count = 0
    count2 = 0
    DarkN = 13.5
    errcount = 0
    exptime = 0
    j = 0
    line_numb = -1
    logg_base = 0
    lspa_vel = 400
    ls2_vel = lspa_vel + 50
    lsrv_vel = 190
    lw2_vel = 15.0
    lwid_vel = 8.0
    M_base = 0
    npix_HER = 4096
    Res_pow = 28000
    res_power_reduction = 0.0
    rvtoll = 800
    rnum = 0
#    shift_lim = 0.0
    SkyN = 0.11
    SNR_add = [1000, 1000, 1000, 1000]
    tnum = 0
    T_base = 0
    vmag = 0
    wavshift = 0
    wavshift2 = 0

#   Strings
    corr_cas = callibration + 'stacked_correct.dat'
    lineparams = callibration + 'lineparams.dat'
    ll = callibration + 'Master_ll'
    rm_location = [callibration + 'Resolving_maps/ccd',
                   '_piv.fits']

#   Arrays
    ap_corr = []
    boun = [[4713, 4901], [5649, 5872], [6478, 6738], [7585, 7885]]
    center_w = [4807.0, 5760.5, 6607.0, 7735.0]
    corr_fun = []
    corr_rv = []
    corr_fun2 = []
    corr_rv2 = []
    corr_fun_tot = np.array([])
    corr_rv_tot = np.array([])
    linecount = [[0, 0], [0, 0], [0, 0], [0, 0]]
    rraise = np.array([])
    sky_band = [-1, -1, -1, -1]
    traise = np.array([])
    wavshift_tot = np.zeros(4)
    prep_flux = [[], [], [], []]
    EP_fin = []
    EWRef = []
    EWR_sig = []
    EWTar = []
    EWT_sig = []
    Li_print = []
    line_rvshift = np.array([])
    R_weight = np.array([])
    R_weighted_average = np.array([])
    Ref_spec = []
    Resolve_R = []
    Resolve_T = []
    resolving_power_map = []
    rspec = []
    rv_correct_wav = [0, 0, 0, 0]
    rv_flag = ['F', 'F', 'F', 'F']
    sky_fits = []
    suc_ele = []
    suc_ion = []
    suc_line = []
    suc_line2 = []
    Tar_spec = []
    tspec = []
    wavshift_vel = [0, 0, 0, 0]

#   Switches
    addSN1 = False
    air_corr = True
    casali_corr = True
    csv = True
    diff = False
    EW_file = False
    harps = False
    line_show = False
    ll_print = False
    ll_test_plot = False
    low_perc = True
    lpp = False
    par = True
    plot_all_band = False
    plot_H = False
    plot_switch = False
    plot_switch1 = False
    plot_switch2 = [False, False, False, False]
    plot_switch3 = False
    plot_switch4 = False
#    plotold = False
    radvelco = True
    ref_measure = False
    resolv_switch = False
    RV_print = True
    sig_mode = False
    single_pivot = True
    sky_flag = False
    SNR_insert = False
    SNR_insert2 = False
    SP_header = False
    SP_header_Li = False
    SP_header_nc = False
    SP_header_rv = False
    stacked = False
    tair_corr = False
    Teff_corr, logg_corr, feh_corr = [], [], []
    test_Li = False
    test_plot_end = False
    warn_flag = True

# Switches (explained in help message)
    for i in range(argc):
        if argv[i] == '--vac' or argv[i] == '-v':
            air_corr = False
        elif argv[i] == '--stacked':
            stacked = True
        elif argv[i] == '--err' or argv[i] == '-e':
            addSN1 = True
            exptime = float(argv[i+1])
            vmag = float(argv[i+2])
        elif argv[i] == '--par':
            lineparams = argv[i+1]
        elif argv[i] == '--self' or argv[i] == '-s':
            stacked = True
            ref_measure = True
            rv_flag_tot = 'FFFF'
        elif argv[i] == '--harps' or argv[i] == '-h':
            harps = True
            ref_measure = True
            stacked = True
            ll = callibration + 'Master_ll_HARPS'
        elif argv[i] == '--harps_ll' or argv[i] == '-l':
            ll = callibration + 'Master_ll_HARPS'
        elif argv[i] == '--sky':
            sky_flag = True
            sky_fits.append(argv[i+1])
        elif argv[i] == '--high_perc':
            low_perc = False
        elif argv[i] == '--casali':
            casali_corr = False
        elif argv[i] == '--ll_print':
            ll_print = True
        elif argv[i] == '--line_prep_plot':
            lpp = True
            line_numb = int(argv[i+1])
        elif argv[i] == '--EW_only':
            EW_file = True
        elif argv[i] == '--air_corr':
            tair_corr = True
        elif argv[i] == '--multi_pivot' or argv[i] == '-m':
            single_pivot = False
        elif argv[i] == '--just_print':
            print(spec_name[5:])
            return
        elif argv[i] == '--SNR_simple':
            SNR_insert = True
            SNR_add = np.ones(4) * float(argv[i+1])
        elif argv[i] == '--SNR':
            SNR_insert = True
            SNR_add[0] = float(argv[i+1])
            SNR_add[1] = float(argv[i+2])
            SNR_add[2] = float(argv[i+3])
            SNR_add[3] = float(argv[i+4])
        elif argv[i] == '--SNR_err_only':
            SNR_insert2 = True
            SNR_add[0] = float(argv[i+1])
            SNR_add[1] = float(argv[i+2])
            SNR_add[2] = float(argv[i+3])
            SNR_add[3] = float(argv[i+4])
        elif argv[i] == '--alt_sig':
            sig_mode = True
        elif argv[i] == '--resolv':
            resolv_switch = True

        if spec_name.split('/')[-1].startswith('Norm_'):
            out_name = spec_name.split('/')[-1].replace('Norm_', '')
        elif spec_name.split('/')[-1].startswith('HERMES_'):
            out_name = spec_name.split('/')[-1].replace('HERMES_', '')
        else:
            out_name = spec_name.split('/')[-1]

    if casali_corr is True:
        with open(corr_cas) as data:
            for line in data:
                if line.startswith('#'):
                    continue
                elif line.split()[0] == 'Temperature_params':
                    Teff_corr = [float(i) for i in line.split()[1:]]
                elif line.split()[0] == 'Surface_gravity_params':
                    logg_corr = [float(i) for i in line.split()[1:]]
                elif line.split()[0] == 'Metallicity_params':
                    feh_corr = [float(i) for i in line.split()[1:]]

    if EW_file is True:
        EW_only(spec_name, lineparams, casali_corr, Teff_corr, logg_corr,
                feh_corr)
        sys.exit(4)

    if sky_flag is True:
        sky_w, sky_f = [0] * len(sky_fits), [0] * len(sky_fits)
        for i in range(len(sky_fits)):
            sky_w[i], sky_f[i] = read_sky(sky_fits[i])
            for j in range(4):
                if any([boun[j][0] < wave < boun[j][1] for wave in sky_w[i]]):
                    sky_band[j] = i
                    break

#   Name setup
    out = spec_name.split('/')[-1] + "EW.dat.gz"
    Resolv_out = spec_name.split('/')[-1] + "Res.dat"

    for i in range(4):
        Ref_spec.append(ref_name + str(i+1) + '.fits')
        Tar_spec.append(spec_name + str(i+1) + '.fits')
        if i == 0:
            if stacked is False:
                if single_pivot is True:
                    try:
                        pivot = int(spec_name[-3:])
                        if pivot > 400:
                            print('WARNING: pivot ID over 400 might not be ' +
                                  'correct.')
                            pivot = 400
                        ap_number[0] = pivot_to_ap(pivot)
                    except ValueError:
                        print(spec_name[-3:])
                        stacked = True
                        print('Warning: No apperature number is given.')
                else:
                    spec_id = spec_name[5:]
                    ap_array = [[], [], [], []]
                    ap_weight_array = [[], [], [], []]
                    for i in range(1, 5):
                        ap_array[i-1], ap_weight_array[i-1] = \
                            combine_ap(callibration + 'ncomb' + str(int(i)) +
                                       '_info.csv', spec_id)
                    if len(ap_array[0]) == 0:
                        print(spec_id)
                        return
        if Path(Tar_spec[-1]).is_file():
            csv = False
    for i in range(4):
        if csv is True:
            Tar_spec[i] = Tar_spec[i][0:-5] + '.csv'

# read the linelist and correct to vacuum if neccessary
    lines, elem, ion, EP = readlinelistw(ll)
    og_lines = lines + 0

#    use = np.zeros(len(lines))
    if air_corr is True:
        lines2 = air2vacESO(lines * u.Angstrom).value

    lines[ion != 3] = lines2[ion != 3]
    out_file = gzip.open(out, mode="wt")
    out_file.write('#elem  ion  wave          EW_ref      sig_EW_ref '
                   'EW_tar      sig_EW_tar EW_difference '
                   'sig_EW_difference\n')

#    out_file2 = open(out2, 'w')

    if single_pivot is True:
        ap_corr = read_ap_correct(lines, elem, ion, ap_number, callibration,
                                  ap_weight)
    else:
        b_ex = [[4000, 8000], [4000, 5000], [5000, 6000], [6000, 7000],
                [7000, 8000]]
        for i in range(1, 5):
            cond_ap = np.bitwise_and(lines > b_ex[i][0], lines < b_ex[i][1])
            lines_temp = lines[cond_ap]
            elem_temp = elem[cond_ap]
            ion_temp = ion[cond_ap]
            ap_corr_temp = read_ap_correct(lines_temp, elem_temp, ion_temp,
                                           ap_array[i-1], callibration,
                                           ap_weight_array[i-1], band=i)
            ap_corr = np.append(ap_corr, ap_corr_temp)

# try to read all the HERMES spectrum bands
    for i in range(4):
        try:
            if stacked is False:
                rspec.append(rHERMES(Ref_spec[i], datahdu=4, SN=True))
            else:
                rspec.append(rHERMES(Ref_spec[i], datahdu=5, SN=True, e_hdu=6))
        except FileNotFoundError:
            rspec.append(0)
            print('Reference band ', i+1, ' not found, skipping it.')
        try:
            tspec.append(rHERMES(Tar_spec[i], datahdu=4, SN=True,
                                 plot_sky=False))
            if tair_corr is True:
                tspec[-1]['w'] = air2vacESO(tspec[-1]['w'] * u.Angstrom).value
            if ref_measure is True:
                twav_temp = np.linspace(boun[i][0], boun[i][1], num=npix_HER)
                tspec[-1]['f'] = prepare_reference_rv(tspec[-1]['w'],
                                                      tspec[-1]['f'],
                                                      twav_temp, Res_pow,
                                                      center_w[i],
                                                      stacked=stacked,
                                                      harps=harps, test=False)

                tspec[-1]['e'] = spectres(twav_temp, tspec[-1]['w'],
                                          tspec[-1]['e'])
                conerr = tspec[-1]['e'] == 0
                tspec[-1]['e'] = np.where([e == 0 for e in tspec[-1]['e']],
                                          0.0025, tspec[-1]['e'])
                tspec[-1]['e'][conerr] = np.multiply(
                        tspec[-1]['e'][conerr],
                        np.sqrt(tspec[-1]['f'][conerr]))
                tspec[-1]['w'] = twav_temp
                tspec[-1]['rv_weight'] = np.ones_like(tspec[-1]['w'])

            if SNR_insert is True:
                results = addSN_simple(tspec[i]['f'], SNR_add[i],
                                       tspec[i]['f'])
                tspec[i]['f'] = results['f']
                tspec[i]['e'] = results['e']
                tspec[i]['SN'] = results['SN']
            if SNR_insert2 is True:
                results = addSN_simple(tspec[i]['f'], SNR_add[i],
                                       tspec[i]['f'])
                tspec[i]['e'] = results['e']
                tspec[i]['SN'] = results['SN']

        except FileNotFoundError:
            tspec.append(0)
            print('Target band ', i+1, ' not found, skipping it.')
        if stacked is False:
            map_path = rm_location[0] + str(i+1) + rm_location[1]
            if single_pivot is True:
                resolving_power_map.append(r_resolving_map(
                        map_path, ap_number, warn_flag=warn_flag))
            else:
                resolving_power_map.append(r_resolving_map(
                        map_path, ap_array[i], warn_flag=warn_flag,
                        weight=ap_weight_array[i]))
            warn_flag = False

        if radvelco is True and tspec[-1] != 0 and rspec[-1] != 0:
            if i == 0:
                tind = int(len(tspec[-1]['f']) * 0.5)
            elif i == 1:
                tind = int(len(tspec[-1]['f']) * 0.512)
            elif i == 2:
                tind = int(len(tspec[-1]['f']) * 0.74)
            elif i == 3:
                tind = int(len(tspec[-1]['f']) * 0.777)
            if stacked is False:
                res_c = resolving_power_map[-1]['w'] == \
                    np.round(tspec[-1]['w'][tind], 1)
                Res_pow = resolving_power_map[-1]['R'][res_c][0]
            rv_correct_wav[i] = tspec[-1]['w'][tind]
#            rdif = rspec[-1]['w'][1] - rspec[-1]['w'][0]
#            rind = np.argwhere(np.bitwise_and(
#                    rspec[-1]['w'] >= tspec[-1]['w'][tind] - rdif/2,
#                    rspec[-1]['w'] < tspec[-1]['w'][tind] + rdif/2))[0][0]

            tdif = (tspec[-1]['w'][-1] - tspec[-1]['w'][0]) / \
                (len(tspec[-1]['w']) - 1)
            inddi = int(tspec[-1]['w'][tind] * rvtoll / c.value / tdif)

            tarflux = tspec[-1]['f'][tind-2*inddi:tind+2*inddi]
            rvweight = tspec[-1]['rv_weight'][tind-2*inddi:tind+2*inddi]
#            tarwave = tspec[-1]['w'][tind-2*inddi:tind+2*inddi]
            tarerr = tspec[-1]['e'][tind-2*inddi:tind+2*inddi]

# Eliminate outliers in the target spectrum
            tarmax = np.max([1.1, 1+5*np.average(tarerr)])
            tarflux = np.where([t > tarmax for t in tarflux],
                               tarmax, tarflux)
            mid_wav = tspec[-1]['w'][tind]

# Reference spectrum preparation
            refwave = tspec[-1]['w'][tind-inddi:tind+inddi]
            refflux = prepare_reference_rv(rspec[-1]['w'], rspec[-1]['f'],
                                           refwave, Res_pow, center_w[i],
                                           stacked=stacked)

            wavshift, corr_rv_temp, corr_fun_temp = determine_radvel(
                    1.0-refflux, 1.0-tarflux, tdif,
                    rvweight, plot_correlation=False, band=i,
                    mid_wav=mid_wav)

            corr_rv.append(corr_rv_temp)
            corr_fun.append(corr_fun_temp)

            tspec[-1]['w'] = tspec[-1]['w'] - wavshift

            wavshift_tot[i] = wavshift

            prep_flux[i] = prepare_reference(rspec[-1]['w'], rspec[-1]['f'],
                                             Res_pow, stacked=stacked)
            if plot_switch1 is True:
                plt.plot(tspec[-1]['w'], tspec[-1]['f'], color='red')
                plt.plot(rspec[-1]['w'], rspec[-1]['f'], color='blue')
                plt.show()
                plt.close()
        else:
            corr_rv.append([0])
            corr_fun.append([0])
#                plt.plot(refwave, refflux, color='blue')
#                plt.plot(rspec[-1]['w'], rspec[-1]['f'], color='red')
#                plt.show()
#                plt.close()

#                plt.plot(tspec[-1]['w'] + wavshift, tspec[-1]['f'] + 0.4,
#                         color='green', linestyle='--')

# Give correlation functions the same rv grid and find the maximum
    o = 0
    for i in range(4):
        if tspec[i] == 0:
            if i == o:
                o = o + 1
            continue
        corr_fun2.append(spectres(corr_rv[o][5:-5], corr_rv[i], corr_fun[i]))
        corr_rv2.append(corr_rv[i])
    corr_fun_tot = np.sum(corr_fun2, 0)
    corr_rv_tot = corr_rv2[0][5:-5]
#    plt.plot(corr_rv_tot, corr_fun_tot)
#    plt.show()
    wavshift_ideal_rv = corr_rv_tot[np.argmax(corr_fun_tot)]

# If the infrared correction failed, we need a backup rv correction
    if ref_measure is False or harps is True:
        for i in range(len(wavshift_vel)):
            if tspec[i] == 0:
                continue
            wavshift_vel[i] = wavshift_tot[i] / rv_correct_wav[i] * c.value
            if np.abs(wavshift_vel[i] - wavshift_ideal_rv) > 10 and \
                    tspec[i] != 0:
                rv_flag[i] = 'T'
                ideal_wavshift = (wavshift_ideal_rv - wavshift_vel[i]) * \
                    rv_correct_wav[i] / c.value
#                print(ideal_wavshift)
                tspec[i]['w'] = tspec[i]['w'] - ideal_wavshift
                wavshift_vel[i] = wavshift_ideal_rv
#        if np.abs(wavshift_vel[0] - wavshift_vel[1]) > 10 and tspec[0] != 0:
#            rv_flag[0] = 'T'
#            tspec[0]['w'] = tspec[0]['w'] + wavshift_tot[0] - \
#                (wavshift_tot[1] * rv_correct_wav[0] / rv_correct_wav[1])
#            wavshift_vel[0] = wavshift_vel[1]
#        if np.abs(wavshift_vel[2] - wavshift_vel[1]) > 10 and tspec[2] != 0:
#            rv_flag[2] = 'T'
#            tspec[2]['w'] = tspec[2]['w'] + wavshift_tot[2] - \
#                (wavshift_tot[1] * rv_correct_wav[2] / rv_correct_wav[1])
#            wavshift_vel[2] = wavshift_vel[1]
#        if np.abs(wavshift_vel[3] - wavshift_vel[1]) > 10 and tspec[3] != 0:
#            rv_flag[3] = 'T'
#            tspec[3]['w'] = tspec[3]['w'] + wavshift_tot[3] - \
#                (wavshift_tot[1] * rv_correct_wav[3] / rv_correct_wav[1])
#            wavshift_vel[3] = wavshift_vel[1]
        rv_flag_tot = rv_flag[0] + rv_flag[1] + rv_flag[2] + rv_flag[3]

    if addSN1 is True:
        for i in range(len(tspec)):
            results = addSN(tspec[i]['f'], exptime,
                            vmag, DarkN, SkyN, i, tspec[i]['f'])
            tspec[i]['f'] = results['f']
            tspec[i]['e'] = results['e']
            tspec[i]['SN'] = results['SN']

    if RV_print is True:
        if not os.path.isfile('rv_statistics.dat'):
            SP_header_rv = True
        with open('rv_statistics.dat', 'a') as rv_write:
            if SP_header_rv is True:
                rv_write.write('# ID                    RV_B    RV_V   ' +
                               'RV_R    RV_IR         \n')
            rv_write.write(str(out_name) + '    ' + str(wavshift_vel[0]) +
                           '    ' + str(wavshift_vel[1]) + '    ' +
                           str(wavshift_vel[2]) + '    ' +
                           str(wavshift_vel[3]) + '\n')

    if line_show is True:
        if csv is False:
            bugfixpdf = matplotlib.backends.backend_pdf.PdfPages(
                    Tar_spec[0].split('/')[-1].rstrip("1.fits") + '_lines.pdf')
        else:
            bugfixpdf = matplotlib.backends.backend_pdf.PdfPages(
                    Tar_spec[0].split('/')[-1].rstrip("1.csv") + '_lines.pdf')

    if ll_print is True:
        nsh = int(len(lines)/4 + 1)
        p_lines = np.round(np.resize(lines, (4, nsh)), decimals=2)
        p_elem = np.resize(elem, (4, nsh))
        p_ion = np.resize(ion, (4, nsh))
        with open('Linelist_write', 'w+') as p_ll:
            for i in range(len(p_lines[0])):
                p_ll.write('       	' +
                           p_elem[0][i] + str(int(p_ion[0][i])) +
                           ' & ' + str(p_lines[0][i]) + ' & ' +
                           p_elem[1][i] + str(int(p_ion[1][i])) +
                           ' & ' + str(p_lines[1][i]) + ' & ' +
                           p_elem[2][i] + str(int(p_ion[2][i])) +
                           ' & ' + str(p_lines[2][i]) + ' & ' +
                           p_elem[3][i] + str(int(p_ion[3][i])) +
                           ' & ' + str(p_lines[3][i]) + ' & ' +
                           ' \\\\' + '\n')

    if plot_H is True:
        plt.plot(tspec[0]['w'], tspec[0]['f'])
        plt.plot(rspec[0]['w'], rspec[0]['f'])
        plt.plot(tspec[2]['w'], tspec[2]['f'])
        plt.plot(rspec[2]['w'], rspec[2]['f'])
        plt.axvline(4861.35, color='black')
        plt.axvline(6562.79, color='black')
        plt.show()
        plt.clf()
    if plot_all_band is True:
        for i in range(4):
            plt.plot(tspec[i]['w'], tspec[i]['f'], color='blue')
            plt.plot(rspec[i]['w'], rspec[i]['f'], color='orange')
        for line in lines:
            plt.axvline(line, color='black')
        plt.show()
        plt.clf()
#    print((tspec[2]['w'][-1] - tspec[2]['w'][0]) / len(tspec[2]['w']))
    j = 0
#    Work on every line separatly
    for line, og_line, ele, io, ep, ap_c in zip(lines, og_lines, elem, ion,
                                                EP, ap_corr):
        rnum, tnum = -1, -1
#        Find the band in which the line is found
#        in target and reference spectra
        for i in range(4):
            if rspec[i] == 0:
                continue
            if rspec[i]['w'][0] < line and line < rspec[i]['w'][-1]:
                rwav = rspec[i]['w']
                rflux = rspec[i]['f']
                rerr = rspec[i]['e']
#                rsnr = rspec[i]['SNR']
                r_prep_f = prep_flux[i]
                rdisp = rwav[1] - rwav[0]
                rnum = i
                break
        for i in range(4):
            if tspec[i] == 0:
                continue
            if tspec[i]['w'][0] < line and line < tspec[i]['w'][-1]:
                twav = tspec[i]['w']
                tflux = tspec[i]['f']
                terr = tspec[i]['e']
#                tsnr = tspec[i]['SNR']
                rv_weight = tspec[i]['rv_weight']
                tdisp = twav[1] - twav[0]
                tnum = i
                break
# If one of the spectra is not available, skip the line
        if rnum == -1 or tnum == -1:
            continue
        if stacked is False:
            res_c = resolving_power_map[tnum]['w'] == \
                np.round(og_line + wavshift_tot[tnum], 1)
            Res_pow = resolving_power_map[tnum]['R'][res_c][0]

        lwid_wav = line * lwid_vel / c.value
        lw2_wav = line * lw2_vel / c.value
        lspa_wav = line * lspa_vel / c.value
        ls2_wav = line * ls2_vel / c.value
        lsrv_wav = line * lsrv_vel / c.value

        if line - lsrv_wav < twav[0] or twav[-1] < line + lsrv_wav:
            continue

        conrv = np.bitwise_and(line - lsrv_wav < rwav,
                               rwav < line + lsrv_wav)

#       Check if the line is close to the edge of the spectrum
        if max(rwav[conrv]) >= max(twav)-0.5 or \
           min(rwav[conrv]) <= min(twav)+0.5:
            continue

#        t_prep_f = prepare_target_wavelegnth(rwav[conrv], twav, tflux)
        t_prep_f = spectres(rwav[conrv], twav, tflux)

        precise_weigth = np.ones_like(t_prep_f)
        for k in range(len(rwav[conrv])):
            index_l = find_nearest_idx(twav, rwav[conrv][k])
            if rv_weight[index_l] == 0:
                precise_weigth[k] = 0

        wavshift2 = determine_radvel(1 - r_prep_f[conrv], 1 - t_prep_f, rdisp,
                                     precise_weigth, mpix=0.2, secondary=True)

        wavshift2_pix = wavshift2 / ((rwav[-1] - rwav[0]) / len(rwav))
        if wavshift2 is False:
            continue

        if plot_switch4 is True and ele == 'Li':
            plt.plot(twav - wavshift, tflux, color='green',
                     linestyle='--')
            plt.plot(twav, tflux+0.4, color='red')
            plt.plot(rwav, rflux, color='blue')
            plt.axhline(1, color='black')
            plt.axvline(line, color='black')
            plt.show()
            plt.close()

        twav = twav - wavshift2

        con = np.bitwise_and(line - lwid_wav < rwav, rwav < line + lwid_wav)

        try:
            nline = center_line(rwav[con], rflux[con])
        except IndexError or ValueError:
            continue

        if lpp is True and j == line_numb:
            lpp_center_w = np.array(rwav)
            lpp_center_f = np.array(rflux)
            lpp_linew = float(nline)
            lpp_linew_old = float(line)
            lpp_window = float(lwid_wav * 4)

        R_ref = 0

        twavcon = np.bitwise_and(nline - ls2_wav < twav,
                                 twav < nline + ls2_wav)
        rwavcon = np.bitwise_and(nline - (ls2_wav * 1.2) < rwav,
                                 rwav < nline + (ls2_wav * 1.2))

        rwav2 = twav[twavcon]
        rflux2 = prepare_reference_rv(rwav[rwavcon], rflux[rwavcon], rwav2,
                                      Res_pow, center_w[rnum], stacked=stacked)
        rerr2 = spectres(rwav2, rwav, rerr)

        if lpp is True and j == line_numb:
            lpp_post_resolv_w = np.array(rwav2)
            lpp_post_resolv_f = np.array(rflux2)
            lpp_target_resolv_w = np.array(twav[twavcon])
            lpp_target_resolv_f = np.array(tflux[twavcon])

        con3 = np.bitwise_and(np.bitwise_and(line - lspa_wav < rwav2,
                                             rwav2 < line + lspa_wav),
                              np.bitwise_not(np.bitwise_and(
                                      line - 2*lwid_wav < rwav2,
                                      rwav2 < line + 2*lwid_wav)))
        con4 = np.bitwise_and(np.bitwise_and(line - lspa_wav < twav,
                                             twav < line + lspa_wav),
                              np.bitwise_not(np.bitwise_and(
                                      line - 2*lwid_wav < twav,
                                      twav < line + 2*lwid_wav)))

        rraise = np.append(rraise, renorm(rflux2[con3]))

        rflux2 = np.multiply(rflux2, rraise[-1])

#        Is the line at least lspa away from the edge of each spectrum?
        if min([line - rwav2[0], rwav2[-1] - line, line -
                twav[0], twav[-1] - line]) < lspa_wav:
            continue

        con = np.bitwise_and(line - lwid_wav*2 < rwav2,
                             rwav2 < line + lwid_wav*2)
        con2 = np.bitwise_and(line - lwid_wav*2 < twav,
                              twav < line + lwid_wav*2)

        if resolv_switch is True and rnum == 2:
            R_ref = determine_resolving_power(rwav2[con], rflux2[con])
            R_tar = determine_resolving_power(twav[con2], tflux[con2])
            if R_ref != -1 and R_tar != -1:
                R_weighted_average = np.append(R_weighted_average,
                                               (R_tar - R_ref))
                R_weight = np.append(R_weight, R_ref)
                with open(Resolv_out, 'a+') as res_out:
                    res_out.write(str(ele) + '    ' + str(line) + '    ' +
                                  str(R_ref) + '    ' + str(R_tar) + '    ' +
                                  str(R_ref - R_tar) + '\n')

# Giving minimal errors
        terr = np.where([e != e for e in terr], -1, terr)
        if stacked is False:
            terr = terr * tflux
        conerr = rerr2 == 0
        try:
            max_terr = max(terr[terr > 0])
            terr = np.where([e <= 0 for e in terr], max_terr, terr)
        except ValueError:
            terr = np.where([e <= 0 for e in terr], 0.25, terr)
        rerr2 = np.where([e == 0 for e in rerr2], 0.0025, rerr2)
        rerr2[conerr] = np.multiply(rerr2[conerr], np.sqrt(rflux2[conerr]))

        if all(rv_weight[con4] == 0):
            continue

        traise_new = lineup(rflux2[con3], tflux[con4], rerr2[con3], terr[con4],
                            band=i, low_perc=low_perc,
                            rv_weight=rv_weight[con4])
        if traise_new is False:
            continue
        traise = np.append(traise, traise_new)
#        print(np.average(traise))
#
        tflux = np.multiply(tflux, traise[-1])
        terr = np.multiply(terr, traise[-1])

        if lpp is True and j == line_numb:
            lpp_post_norm_w = np.array(twav)
            lpp_post_norm_f = np.array(tflux)
            lpp_reference_norm_f = np.array(rflux2)
            lpp_reference_norm_w = np.array(rwav2)

        if plot_switch2[tnum]:
            plt.plot(rwav2, rflux2, label='Ref spectrum')
            plt.plot(twav, tflux, label='Tar spectrum')
            plt.plot(twav, terr/(np.max(terr)*2), label='Err Array')
            plt.axhline(1.0, color='black')
            plt.axvline(line, color='blue')
            plt.axvline(nline, color='purple', linestyle='--')
            plt.xlim(line - lspa_wav, line + lspa_wav)
            plt.legend(loc='lower left')
            plt.show()
            plt.clf()

        if ll_test_plot is True:
            plt.title(ele + ' ' + str(line))
            plt.plot(rwav2, rflux2, label='Ref spectrum')
            plt.plot(twav, tflux, label='Tar spectrum')
            plt.axhline(1.0, color='black')
            plt.axvline(line - lw2_wav, color='blue', ls='--')
            plt.axvline(line + lw2_wav, color='blue', ls='--')
            plt.axvline(line, color='blue')
            plt.axvline(nline, color='purple', linestyle='--')
            plt.xlim(line - lspa_wav, line + lspa_wav)
            plt.legend(loc='lower left')
            plt.show()
            plt.clf()

        if plot_switch3 is True and j == 40:
            pdf = matplotlib.backends.backend_pdf.PdfPages("Line_measure.pdf")
            fig, p = plt.subplots(1, 1)
            fig.set_figheight(6)
            fig.set_figwidth(11)

            p.step(rwav2, rflux2, label='Ref spectrum')
            p.step(twav, tflux, label='Tar spectrum')
            p.axhline(1, color='black', ls='--')
            p.axvline(nline - lw2_wav, color='blue', ls='--')
            p.axvline(nline + lw2_wav, color='blue', ls='--')
            p.axvline(nline, color='purple', linestyle='-')
            p.set_xlim(line - lspa_wav/2, line + lspa_wav/2)
            p.legend(loc='lower right')
            p.set_xlabel(r'\LARGE Wavelength [\AA]')
            p.set_ylabel(r'\LARGE Normalized Flux')

            p.xaxis.set_minor_locator(AutoMinorLocator())
            p.yaxis.set_minor_locator(AutoMinorLocator())
            p.set_rasterization_zorder(-20)
            pdf.savefig(fig)
            plt.clf()
            pdf.close()

        if nline < 0:
            if nline == -2:
                errcount = errcount + 1
            else:
                count = count + 1
#            print(ele + '    ' + str(io) + '    ' + str(line) + '    ' + '5')
            continue

# weight the line by the errors of pixel and how much they pixel
# fall into the wavelength region
# We assume an equal pixel spacing in this step which is normally provided
# by HERMES itself
        tweight, tnpix = add_weight(nline, lw2_wav, twav, terr, tdisp)

        if np.sum(tweight) > np.sum(np.multiply(tweight, rv_weight)):
            continue

        if plot_switch is True:
            terr2 = np.absolute(np.power(tflux[twavcon], 0.5))
            plt.plot(twav, tweight/(max(tweight)*2), color='red')
            plt.plot(twav[twavcon], terr2/(max(terr2)*2), color='blue',
                     linestyle='--')
            plt.plot(twav, terr/(max(terr[twavcon])*2), color='red',
                     linestyle='--')
            plt.plot(rwav2, rflux2, label='Sun spectrum')
            plt.plot(twav, tflux, label='Potential Solar Twin')
            plt.axhline(1.0, color='black')
            plt.axvline(nline, color='purple', linestyle='--')
            plt.xlim(line - 2, line + 2)
            plt.ylim(-0.1, 1.5)
            plt.legend(loc='lower left')
            plt.show()
            plt.clf()

#       Measure the EW of the line
#       These need to be changed if EPIC is to be used on an instrument with
#       non-equal pixel spacing
        rEW, rEW_sig = measure_EW(rwav2, rflux2, rerr2, tweight[twavcon],
                                  lw2_wav)
        tEW, tEW_sig = measure_EW(twav, tflux, terr, tweight, lw2_wav)

        if stacked is False:
            rEW = rEW / ap_c
            tEW = tEW / ap_c
            rEW_sig = rEW_sig / ap_c
            tEW_sig = tEW_sig / ap_c

        EW_full_err = np.sqrt(np.square(rEW_sig) + np.square(tEW_sig))

        if lpp is True and j == line_numb:
            lpp_weights = np.array(tweight[twavcon])
            lpp_twavcon = twavcon

        if (rEW < 5 or tEW < 5) and int(line) != 6709:
            count2 = count2+1
            if test_plot_end is True:
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.step(rwav2, rflux2, lw=1, where='mid', color='purple',
                        label='solar')
                ax.step(twav, tflux, lw=1, where='mid', color='orange',
                        label='HERMES')
                ax.axhline(1, color='black', ls='--', lw=1)
                ax.axvline(nline, color='black', ls='-', lw=1)
                ax.axvline(nline - lw2_wav, color='red', ls='--', lw=1)
                ax.axvline(nline + lw2_wav, color='red', ls='--', lw=1)
                ax.set_xlabel(r'wavelength [\AA]')
                ax.set_ylabel(r' Normalized Flux')
                ax.legend(loc='lower left', handlelength=1.0)
                ax.set_rasterization_zorder(-10)
                plt.show()
                ax.clear()
            out_file.write(str(ele).ljust(3, ' ') + '    '
                           + str(int(io)) + '    '
                           + "{0:.4f}".format(line) + '    '
                           + "{0:.4f}".format(rEW).rjust(8, ' ') + '    '
                           + "{0:.4f}".format(rEW_sig).rjust(7, ' ') + '    '
                           + "{0:.4f}".format(tEW).rjust(8, ' ') + '    '
                           + "{0:.4f}".format(tEW_sig).rjust(7, ' ') + '    '
                           + "{0:.4f}".format(rEW - tEW) + '    '
                           + "{0:.4f}".format(EW_full_err) + '    '
                           + "#" + "\n")
            continue
        if rEW*5 < tEW:
            out_file.write(str(ele).ljust(3, ' ') + '    '
                           + str(int(io)) + '    '
                           + "{0:.4f}".format(line) + '    '
                           + "{0:.4f}".format(rEW).rjust(8, ' ') + '    '
                           + "{0:.4f}".format(rEW_sig).rjust(7, ' ') + '    '
                           + "{0:.4f}".format(tEW).rjust(8, ' ') + '    '
                           + "{0:.4f}".format(tEW_sig).rjust(7, ' ') + '    '
                           + "{0:.4f}".format(rEW - tEW) + '    '
                           + "{0:.4f}".format(EW_full_err) + '    '
                           + "#" + "\n")
            continue

        if test_Li is True and ele == 'Li':
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.step(rwav2, rflux2, lw=1, where='mid', color='purple',
                    label='solar')
            ax.step(twav, tflux, lw=1, where='mid', color='orange',
                    label='HERMES')
            ax.axhline(1, color='black', ls='--', lw=1)
            ax.axvline(nline, color='black', ls='-', lw=1)
            ax.axvline(nline - lw2_wav, color='red', ls='--', lw=1)
            ax.axvline(nline + lw2_wav, color='red', ls='--', lw=1)
            ax.set_xlabel(r'wavelength [\AA]')
            ax.set_ylabel(r' Normalized Flux')
#            ax.legend(loc='lower left', handlelength=1.0)
            ax.set_rasterization_zorder(-10)
            plt.show()
            ax.clear()

        if 3 * EW_full_err < rEW - tEW:
            linecount[tnum][1] += 1

        elif -3 * EW_full_err > rEW - tEW:
            linecount[tnum][1] += 1

        else:
            linecount[tnum][0] += 1
#        After processing for the ploting and text file with EW.
        out_file.write(str(ele).ljust(3, ' ') + '    ' + str(int(io))
                       + '    ' + "{0:.4f}".format(line) + '    '
                       + "{0:.4f}".format(rEW).rjust(8, ' ') + '    '
                       + "{0:.4f}".format(rEW_sig).rjust(7, ' ') + '    '
                       + "{0:.4f}".format(tEW).rjust(8, ' ') + '    '
                       + "{0:.4f}".format(tEW_sig).rjust(7, ' ') + '    '
                       + "{0:.4f}".format(rEW - tEW) + '    '
                       + "{0:.4f}".format(EW_full_err) + "\n")
        suc_ele.append(ele)
        suc_ion.append(io)
        suc_line.append(line)
        EWRef.append(rEW)
        EWTar.append(tEW)
        EWR_sig.append(rEW_sig)
        EWT_sig.append(tEW_sig)
        EP_fin.append(ep)
        line_rvshift = np.append(line_rvshift,
                                 (wavshift_tot[tnum] + wavshift2) /
                                 line * c.value)

        if R_ref > 0:
            suc_line2.append(line)
            Resolve_R.append(R_ref)
        j = j+1

        if ele == 'Li':
            Li_print = [rEW, rEW_sig, tEW, tEW_sig, wavshift2_pix]
    if len(R_weight) != 0 and len(R_weighted_average) != 0:
        res_power_reduction = np.sum(R_weighted_average) / np.sum(R_weight)
    else:
        res_power_reduction = 0

    rv_weight2 = np.square(EWTar)
    if np.sum(rv_weight2) <= 0:
        return
    rv_weighted_av = np.average(line_rvshift, weights=rv_weight2)

    if lpp is True:
        line_prep_plot(lpp_center_w, lpp_center_f, lpp_linew, lpp_linew_old,
                       lpp_window, lpp_post_resolv_w, lpp_post_resolv_f,
                       lpp_target_resolv_w, lpp_target_resolv_f,
                       lpp_post_norm_w, lpp_post_norm_f, lpp_reference_norm_w,
                       lpp_reference_norm_f, lpp_weights, lpp_twavcon)

    out_file.close()

    if len(Resolve_T) > 5:
        plt.plot(suc_line2, Resolve_T)
        plt.plot(suc_line2, Resolve_R)
        plt.show()
        plt.close()

    if line_show is True:
        bugfixpdf.close()

#    Calculating the Stellar Parameters using the stacked HERMES spectral data.
    if par is True:
        EW_base = np.array([], dtype='float64')
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
        EW_std = np.array([], dtype='float64')
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
                EW_std = np.append(EW_std, float(lin[19]))
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
                if diff is True:
                    EW_base = np.append(EW_base,
                                        hypersurfacelstsq([a1[i], a2[i], b1[i],
                                                           b2[i], d1[i], d2[i],
                                                           com1[i], off[i]],
                                                          T_base, logg_base,
                                                          M_base))
                    EW2[-1] = EW_base[-1] + EWRef[pos[0][0]] - EWTar[pos[0][0]]
            else:
                sl += 1
        if False:
            print(len(EW2))
        initialParameters = [5750.0, 4.4, 0.1]
#        initial_err = np.abs(np.array([5772.0, 4.438, 0.0])
#                             - np.array(initialParameters))
#        initial_err = initialParameters
        parabound = ([5000, 3.2, -0.8], [6500, 5.0, 0.55])
        try:
            EW_fit_sig = np.sqrt(np.square(EW2_sig) + np.square(EW_std[indic]))
#            print(EW2_sig / EW_fit_sig)
#            print(EW_std[indic] / EW_fit_sig)
            popt, pcov = curve_fit(hypersurfacelstsq, [a1[indic], a2[indic],
                                                       b1[indic], b2[indic],
                                                       d1[indic], d2[indic],
                                                       com1[indic],
                                                       off[indic]], EW2,
                                   p0=initialParameters, sigma=EW_fit_sig,
                                   bounds=parabound, absolute_sigma=sig_mode)

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

        if reduce_out is True:
            if casali_corr is True:
                T_cas = dTemp_lin(lstsq, Teff_corr[0], Teff_corr[1],
                                  Teff_corr[2], Teff_corr[3], Teff_corr[4])
                logg_cas = dlogg_lin(lstsq, logg_corr[0], logg_corr[1],
                                     logg_corr[2], logg_corr[3], logg_corr[4])
                feh_cas = dMetal(lstsq, feh_corr[0], feh_corr[1], feh_corr[2],
                                 feh_corr[3], feh_corr[4])

                lstsq[0] = lstsq[0] - T_cas
                lstsq[1] = lstsq[1] - logg_cas
                lstsq[2] = lstsq[2] - feh_cas

                lstsq_sig[0] = lstsq_sig[0] * (1 - Teff_corr[0])
                lstsq_sig[1] = lstsq_sig[1] * (1 - logg_corr[1])
                lstsq_sig[2] = lstsq_sig[2] * (1 - feh_corr[2])

            if not os.path.isfile('stellar_params_SNR_grid.dat'):
                SP_header = True
            with open('stellar_params_SNR_grid.dat', 'a') as out:
                if SP_header is True:
                    out.write('#      SNR_B       SNR_G       SNR_R       '
                              + 'SNR_IR    T_eff         dT_eff     logg  '
                              + '     dlogg      Fe_H        dFe_H\n')
                out.write(('{:>11.1f} {:>11.1f} {:>11.1f} {:>11.1f} {:>11.1f}'
                           + ' {:>11.1f} {:>11.4f}{:>11.4f} {:>11.5f} '
                           + '{:>11.5f}\n'
                           ).format(SNR_add[0], SNR_add[1], SNR_add[2],
                          SNR_add[3], lstsq[0], lstsq_sig[0], lstsq[1],
                          lstsq_sig[1], lstsq[2], lstsq_sig[2]))
        else:
            if not os.path.isfile('stellar_params_no_casali.dat'):
                SP_header_nc = True
            with open('stellar_params_no_casali.dat', 'a') as out:
                if SP_header_nc is True:
                    out.write('# ID                    T_eff    dT_eff   ' +
                              'logg    dlogg    Fe_H      dFe_H    RV    ' +
                              '     \n')
                out.write(('{:>5s} {:>11.1f} {:>11.1f} {:>11.4f}'
                          + '{:>11.4f} {:>11.5f} {:>11.5f}\n').format(
                          out_name, lstsq[0],
                          lstsq_sig[0], lstsq[1], lstsq_sig[1], lstsq[2],
                          lstsq_sig[2]))
            if not os.path.isfile('Li_line.dat'):
                SP_header_Li = True
            if len(Li_print) == 5:
                with open('Li_line.dat', 'a') as out:
                    if SP_header_Li is True:
                        out.write('# ID                    rEW    rEW_sig   ' +
                                  'tEW    tEW_sig    RV_pix     \n')
                    out.write(('{:>5s} {:>11.4f} {:>11.4f} {:>11.4f}'
                              + '{:>11.4f} {:>11.4f}\n').format(
                              out_name, Li_print[0], Li_print[1], Li_print[2],
                              Li_print[3], Li_print[4]))

            if casali_corr is True:
                T_cas = dTemp_lin(lstsq, Teff_corr[0], Teff_corr[1],
                                  Teff_corr[2], Teff_corr[3], Teff_corr[4])
                logg_cas = dlogg_lin(lstsq, logg_corr[0], logg_corr[1],
                                     logg_corr[2], logg_corr[3], logg_corr[4])
                feh_cas = dMetal(lstsq, feh_corr[0], feh_corr[1], feh_corr[2],
                                 feh_corr[3], feh_corr[4])
                lstsq[0] = lstsq[0] - T_cas
                lstsq[1] = lstsq[1] - logg_cas
                lstsq[2] = lstsq[2] - feh_cas

                lstsq_sig[0] = lstsq_sig[0] * (1 - Teff_corr[0])
                lstsq_sig[1] = lstsq_sig[1] * (1 - logg_corr[1])
                lstsq_sig[2] = lstsq_sig[2] * (1 - feh_corr[2] -
                                               feh_corr[3] * lstsq[2])

            if not os.path.isfile('stellar_params.dat'):
                SP_header = True
            with open('stellar_params.dat', 'a') as out:
                if SP_header is True:
                    out.write('# ID                    T_eff    dT_eff   ' +
                              'logg    dlogg    Fe_H      dFe_H    RV      ' +
                              'Res_power_red     RV_flag   \n')
                out.write(('{:>5s} {:>11.1f} {:>11.1f} {:>11.4f}'
                          + '{:>11.4f} {:>11.5f} {:>11.5f} {:>11.5f} {:>11.5f}'
                          + ' {:>5s}\n'
                           ).format(out_name, lstsq[0],
                          lstsq_sig[0], lstsq[1], lstsq_sig[1], lstsq[2],
                          lstsq_sig[2], rv_weighted_av, res_power_reduction,
                          rv_flag_tot))
    return 0


if __name__ == '__main__':
    argv = np.array(sys.argv)
    switch = np.zeros_like(argv, dtype=bool)
    spec = '0'
    SNR_start = 10
    SNR_stop = 91
    SNR_step = 5
    SNR_string = "--SNR_err_only"

    a_B = [-7.9221093, 0.73295998, -0.0014636126]
    a_G = [-5.76622, 0.776005, -0.00100738]
    a_IR = [-2.95438, 0.846138, -0.000380174]

    for i, arg in enumerate(argv):
        if arg.endswith('.fits') or arg.endswith('EW.dat.gz'):
            switch[i] = False
        else:
            switch[i] = True
        if arg == "--SNR_grid":
            mode_string = '--alt_sig'
            for SNR_R in np.arange(SNR_start, SNR_stop, SNR_step):
                if spec == '0':
                    spec = argv[np.bitwise_not(switch)][1]
                SNR_B = linear_SNR(a_B, SNR_R)
                SNR_G = linear_SNR(a_G, SNR_R)
                SNR_IR = linear_SNR(a_IR, SNR_R)
                SNR3 = SNR_R
                for SNR1 in np.arange(SNR_B - 5, SNR_B + 6, 5):
                    for SNR2 in np.arange(SNR_G - 5, SNR_G + 6, 5):
                        for SNR4 in np.arange(SNR_IR - 5, SNR_IR + 6, 5):
                            SNR1 = max(SNR1, 0.1)
                            SNR2 = max(SNR2, 0.1)
                            SNR4 = max(SNR4, 0.1)
                            argv2 = np.append([argv[0], SNR_string,
                                               str(SNR1), str(SNR2),
                                               str(SNR3), str(SNR4),
                                               mode_string], argv[1:])
                            switch = np.zeros_like(argv2, dtype=bool)

                            for i, arg in enumerate(argv2):
                                if arg.endswith('.fits') or \
                                  arg.endswith('EW.dat.gz'):
                                    switch[i] = False
                                else:
                                    switch[i] = True
                            main_EPIC(argv=argv2[switch], spec_name=spec,
                                      ref_name=argv2[switch][-1],
                                      reduce_out=True)
            exit(0)

    if len(argv) - len(argv[switch]) == 0:
        main_EPIC()
    else:
        for spec in argv[np.bitwise_not(switch)]:
            main_EPIC(argv=argv[switch], spec_name=spec[:-6],
                      ref_name=argv[switch][-1])
