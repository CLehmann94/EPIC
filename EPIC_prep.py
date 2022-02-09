#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from EPIC_scripts import rHARPS, rflatHARPS, HAR2HER, HAR2HER2, rHERMES_prep, \
                      read_unreduced, normalize_HERMES, wHERMES, wHERMES2


# Help Message
def helpmessage():
    print('')
    print('A program to convert HARPS spectra and convert '
          'them to HERMES spectra. It is also used to normalize '
          'HERMES spectra with the --HERMES option.')
    print('')
    print('Usage: ', PROGNAME, '(Options) [fits files]')
    print('')
    print('Options:  --help          Print this message.')
    print('          --HERMES        Normalize HERMES spectra')
    print('          --unreduced     Prepare a full fibre .fits')
    print('')
    sys.exit(3)


if __name__ == '__main__':
    PROGNAME = sys.argv[0]
    argc = len(sys.argv)


# looking for help?
    if argc <= 1:
        helpmessage()
    else:
        for i in range(argc):
            if sys.argv[i] == '--help':
                helpmessage()


# Some numbers setup
    specres = 28000  # HERMES should have 28000.
    pixelw = 0.01
    exptime = 0
    output = {}
    noop = 1
    SNR = [0, 0, 0, 0]
    SNRpp = [0, 0, 0, 0]
    other_header = False
    bands = 4
    HARPS = True
    datahdu = 0
    mask = False
    plot_cuts = False
    unreduced = False
    HERMESspec = 0
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

#   Switches (explained in help message)
    for i in range(argc):
        if sys.argv[i] == '--res':
            specres = float(sys.argv[i+1])
            pixelw = float(sys.argv[i+2])
            noop += 3
        elif sys.argv[i] == '--head':
            other_header = True
            header_file_temp = sys.argv[i+1]
            noop += 2
        elif sys.argv[i] == '--HERMES':
            HARPS = False
            bands = 1
            noop += 1
        elif sys.argv[i] == '--reduce_mask':
            mask = True
            noop += 1
        elif sys.argv[i] == '--plot_cuts':
            plot_cuts = True
            noop += 1
        elif sys.argv[i] == '--unreduced':
            HARPS = False
            unreduced = True
            noop += 1

    # Are all of those .fits files?
    for i in range(noop, argc):
        if ".fits" not in sys.argv[i] and ".ascii" not in sys.argv[i] and \
                ".csv" not in sys.argv[i]:
            print("error: ", sys.argv[i], " is not a .fits or .ascii file")
            sys.exit(1)


#   Use the sub-scripts (described in the all the respective scripts)
    for i in range(noop, argc):
        print(sys.argv[i])
        if other_header is True:
            header_file = header_file_temp
        else:
            header_file = sys.argv[i]

        if HARPS is True:
            try:
                fitsspec = rHARPS(sys.argv[i], obj=True, SN=True)
                flatflag = 1
                if 'SN' not in fitsspec.keys():
                    fitsspec['SN'] = 1000
            except ValueError:
                fitsspec = rflatHARPS(sys.argv[i], SN=True)
                flatflag = 0
        elif unreduced is True:
            fitsspec = read_unreduced(sys.argv[i])
        else:
            fitsspec = rHERMES_prep(sys.argv[i], datahdu=datahdu)

        if np.max(fitsspec["w"]) < 7885:
            bands = 3

        if HARPS is True:
            HERMESspec = HAR2HER2(fitsspec)
            HERMESspec['norm_sky_f'] = np.ones_like(HERMESspec['f'])
            normalization_spec = HAR2HER(fitsspec, specres, pixelw,
                                         band_cor=False)
            normalization_spec2 = HAR2HER(fitsspec, specres, pixelw)

            for j in range(bands):
                continuum, err = normalize_HERMES(
                        HERMESspec['w'][j],
                        HERMESspec['f'][j], HERMESspec['w'][j],
                        Kurucz=mask, plot_cuts=plot_cuts)

                HERMESspec['e'][j] = err
                HERMESspec['norm_f'][j] = HERMESspec['f'][j] / continuum

                con = np.greater_equal(HERMESspec['norm_f'][j],
                                       1.03 + HERMESspec['e'][j]*3)
                for k in range(len(con)):
                    if con[k] is True:
                        con[k-2] = True
                        con[k-1] = True
                        con[k+1] = True
                        con[k+2] = True

                HERMESspec['norm_sky_f'][j] = np.where(con, -1,
                                                       HERMESspec['norm_f'][j])
        elif unreduced is True:
            for l in range(len(fitsspec['f'])):
                if str(fitsspec['name'][l]).startswith('Sky') or \
                  str(fitsspec['name'][l]).startswith('FIBRE'):
                    print('Sky fibre or fibre not in use.')
                    continue
                if len(fitsspec['e'][l][fitsspec['e'][l] > 0]) == 0:
                    print('Warning: {:>5s} does not contain flux.'.format(
                            fitsspec['name'][l]))
                    continue
                SNR = np.nanmedian(fitsspec['f'][l]) / \
                    np.nanmedian(fitsspec['e'][l])
                print(fitsspec['name'][l], SNR)
                continuum, _ = \
                    normalize_HERMES(fitsspec['w'], fitsspec['f'][l],
                                     fitsspec['w'], e=fitsspec['e'][l],
                                     Kurucz=mask, plot_cuts=plot_cuts)
                fitsspec['norm_f'][l] = fitsspec['f'][l] / continuum
                fitsspec['norm_f'][l][fitsspec['norm_f'][l] !=
                                      fitsspec['norm_f'][l]] = np.ones_like(
                         fitsspec['norm_f'][l][fitsspec['norm_f'][l] !=
                                               fitsspec['norm_f'][l]])
                fitsspec['e'][l] = fitsspec['e'][l] / continuum

                sky_continuum, _ = normalize_HERMES(
                    fitsspec['w'], fitsspec['sky_f'][l], fitsspec['w'],
                    e=fitsspec['e'][l], Kurucz=mask, plot_cuts=False)
                fitsspec['norm_sky_f'][l] = fitsspec['sky_f'][l] / \
                    sky_continuum

                con = np.greater_equal(fitsspec['norm_sky_f'][l],
                                       1.03 + fitsspec['e'][l]*3)

                for j in range(len(con)):
                    if con[j] is True:
                        con[j-2] = True
                        con[j-1] = True
                        con[j+1] = True
                        con[j+2] = True

                fitsspec['norm_sky_f'][l] = np.where(con, -1,
                                                     fitsspec['norm_sky_f'][l])

        else:
            title = sys.argv[i][5:-7]
            HERMESspec = fitsspec
            continuum, _ = normalize_HERMES(HERMESspec['w'], HERMESspec['f'],
                                            HERMESspec['w'], e=HERMESspec['e'],
                                            title=title, Kurucz=mask,
                                            plot_cuts=plot_cuts)
            if continuum[0] == -1:
                continue
            HERMESspec['norm_f'] = HERMESspec['f'] / continuum
            HERMESspec['e'] = HERMESspec['e'] / continuum
            if 'sky_f' in HERMESspec:
                sky_continuum, _ = normalize_HERMES(
                        HERMESspec['w'], HERMESspec['sky_f'], HERMESspec['w'],
                        e=HERMESspec['e'], title=title, Kurucz=mask,
                        plot_cuts=plot_cuts)
                HERMESspec['norm_sky_f'] = HERMESspec['sky_f'] / \
                    sky_continuum

                con = np.greater_equal(HERMESspec['norm_sky_f'],
                                       1.03 + HERMESspec['e']*3)
                for j in range(len(con)):
                    if con[j] is True:
                        con[j-2] = True
                        con[j-1] = True
                        con[j+1] = True
                        con[j+2] = True

                HERMESspec['norm_sky_f'] = np.where(con, -1,
                                                    HERMESspec['norm_sky_f'])
                con_low = np.less_equal(HERMESspec['f'], 0.1)
                HERMESspec['norm_sky_f'] = np.where(con_low, -1,
                                                    HERMESspec['norm_sky_f'])

        if HERMESspec == 0 and fitsspec == 0:
            print("Error: ", sys.argv[i],
                  " has not the right wavelength range.")
            continue

        if HARPS is True:
            HERMESspec['EXPOSED'] = exptime
            if sys.argv[i].startswith('ADP.'):
                object_name = fitsspec['obj'].strip()
                if object_name.startswith('HIP-'):
                    object_name = 'HIP' + object_name.lstrip('HIP-')
                output[i] = 'HERMES_' + object_name + '.fits'
            else:
                output[i] = "HERMES_" + sys.argv[i].split('/')[-1]
            wHERMES(HERMESspec, header_file, output[i], SNR=SNR,
                    SNRpp=SNRpp, bands=bands)
        elif unreduced is True:
            fitsspec['EXPOSED'] = exptime
            for l in range(len(fitsspec['f'])):
                output[l] = 'Norm_' + str(fitsspec['name'][l]) + \
                    str(fitsspec['band']) + '.fits'
                band = str(fitsspec['band'])
                HERMESspec = {'w': fitsspec['w'], 'f': fitsspec['f'][l],
                              'e': fitsspec['e'][l],
                              'norm_sky_f': fitsspec['norm_sky_f'][l],
                              'norm_f': fitsspec['norm_f'][l],
                              'EXPOSED': fitsspec['EXPOSED']}
                wHERMES2(HERMESspec, header_file, output[l])
        else:
            HERMESspec['EXPOSED'] = exptime
            path = ''
            p_array = sys.argv[i].split('/')[0:-1]
            r_name = sys.argv[i].split('/')[-1]
            for str1 in reversed(p_array):
                path = str1 + '/' + path
            output[i] = path + 'Norm_' + r_name.split('.')[-2] + '.fits'
            band = sys.argv[i].split('.')[-2][-1]
            wHERMES2(HERMESspec, header_file, output[i])
