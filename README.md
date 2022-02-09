# EPIC
The EPIC algorithm was created to analyse spectra from the HERMES spectrograph effectively and efficiently. It calculates equivalent witdths (EWs) for lines found within a line list (default Master_ll5.2) and uses a sub-group of these line in order to calculate stellar parameters (effective temperature T_eff, surface gravity log(g) and metallicity [Fe/H]). The whole process of the algorithm was explained in detail in Lehmann et al. (2022).


Spectrum preparation:
Every spectrum that EPIC may be applied to should be prepared with the 


Usage:
EPICv1.py (Options) [1.reference / 2.target HERMES]

EPIC is simply used via the command line and at the minimum requires two arguments: The reference spectrum and at least one target spectrum.

ex: EPIC_V1.py Reference_spectrum/HERMES_Kurucz example_spectra/Norm_1706150044012332.fits

Note that EPIC will search for the endings 1.fits, 2.fits, 3.fits and 4.fits (for each band of HERMES) itself when one of them is entered.
One might be interested in analysing all spectra within a folder, simply insert them all as the last arguments or use a wildcart command:

EPIC_V1.py Reference_spectrum/HERMES_Kurucz example_spectra/Norm*2.fits
(under Linux)


Options:
EPIC has several option that might be needed in different situations:

-e, --err [exptime] [vmag]: Add noise to the target spectrum by giving exposure time(s), mag of V band and base noise. The noise in each band is calculated using solar relations between the bands, e.g. the red band has r_mag = v_mag - 0.352

-h, --harps: The target spectra are spectra taken with HARPS and simulated to HERMES resolution.

-s, --self: Simplify the calculations within EPIC when using the reference spectrum as a target spectrum (comparing it against itself). 

--SNR [SNR_b] [SNR_v] [SNR_r] [SNR_ir]: Add noise to the the spectrum's flux and modify the error array accordingly via a specific signal-to-noise ratio for each band. The SNRs must be given in the order blue band SNR (spectrum ending in 1.fits), green band SNR (spectrum ending in 2.fits), red band SNR (spectrum ending in 3.fits) and infrared band SNR (spectrum ending in 4.fits).

--SNR_err_only [SNR_b] [SNR_v] [SNR_r] [SNR_ir]: Similar process as in the above command, but only modifies the error array and leaves the flux array untouched. This is to give high SNR spectra (e.g. the solar atlas) a realistic error array that makes it better comparable with real HERMES spectra. in the calibrations we used --SNR_err_only 50 50 50 50 for all HARPS spectra and the solar atlas.

-v, --vac: The spectrum is in vaccuum and does not need to be corrected for it. By default every spectrum gets corrected from air wavelength to vaccuum wavelengths (using the Edlen 1953 formula). This is not adviced when working with HERMES spectra provided by GALAH.




Output:
Each of these files can be read in via Topcat using the Ascii input option.

Li_line.dat: A file that gathers the EW of the Li line at 6707.76 Angstrom. It displys the EW for bother the reference and target spectrum as well as their uncertainties. EPIC will add to this file for each spectrum that is analysed to give a statistic for the Li lines.
Note that this is a weak and not very stable line, so most low SNR spectra will not measure a reasonable value here. It is only interesting to measure this line when the Li content in a star is high.

EW.dat.gz file: A gziped file that contains the EW width measurments of all lines within the line list that was given to EPIC. Each spectrum will have a new file that saves all their respective EW values. Displays EW for both the target and reference spectrum as well as their difference and uncertainties.
Note that not all lines are used to measure the EW, but all lines will have an EW measurement if the measurement can be done successfully.

rv_statistics.dat: A file collecting the radial velocity that has been calculated for each band. If the radial velocity correction has been applied correctly, each of these numbers should be within 5km/s (~1 pixel) of each other.

stellar_params.dat: A file containing the three stellar parameters (effective temperature, surface gravity and metallicity) as well as their uncertainties. In addition there is the average radial velocity between the 4 bands and a radial velocity flag showing if all the radial velocities have been computed correctly (e.g. FFFF is a no band failed with the correction, FTTF means that the V and R band failed their radial velocity correction). Lastly there is the Resolving power reduction which is not implemented completely, yet.

stellar_params_no_casali.dat: The same as stellar_params.dat but the stellar parameters are computed without the higher-order correction that uses the Casali et al. (2020) data. See Lehmann et al. (2022).




