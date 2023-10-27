# rvpipe
## A command-line pipeline for fast line-by-line rv processing of NEID solar spectra.

## Required files: 
- Line temperature table (`"T1o2_spec-2.csv"`)
- Telluric line template (`"TAPAS_WMKO_NORAYLEIGH_SPEC.fits"`, `"TAPAS_WMKO_NORAYLEIGH_WVL.fits"`)

## Arguments:

`python3 rvpipe.py ["target file directory"] [-c] [cpu count] [-t] [telluric mask strength] [-w] [-m] [-n] [-x] [-l] [-d] [-i]`

`["target file directory"]`: Required. Enter directory to target files for processing

`[-c] [cpu count]`: Specify number of cpus to use (default is all available)

`[-t] [telluric mask strength]`: Specify telluric mask strength as a fraction of line depth (default = 4 (1e-4))

`[-w] [minimum wavelength]`: Specify minimum wavelength to use for analysis in angstroms (default = 0)

`[-m] [maximum wavelength]`: Specify minimum wavelength to use for analysis in angstroms (default = 7000 Å)

`[-n] [minimum line width]`: Specify minimum line width in pixels (default = 10)

`[-x] [maximum line width]`: Specify maximum line width in pixels (default = 100)

`[-l] [minimum line depth]`: Specify minimum line depth (default = 0.005)

`[-d] [maximum continuum difference`: Specify maximum continuum difference (default = 0.05)

`[-i]`: Disable automatic creation of reference spectrum (`"refspectrum.npz"`) if one already exists from a previous run

Example: `python3 rvpipe.py 'data' -c 6 -t 5 -i`

## Outputs:

`"all_lines.npz"`: Contains 6 numpy arrays, all in the format `[file, line]`.
- arr_0: rvs for each file and line
- arr_1: rv errors
- arr_2: correlation coefficients between reference spectrum and line
- arr_3: FWHM of each line
- arr_4: pixel width of each line
- arr_5: line depth of each line

`"output_file.npz"`: Contains 14 numpy arrays.
- arr_0: mean RV for each file
- arr_1: RV error for each file
- arr_2: mean RV from the NEID pipeline for each file
- arr_3: time (Julian Date)
- arr_4: solar altitude (deg)
- arr_5: wavelength (Angstroms) for each detected line
- arr_6: continuum difference for each detected line, from reference spectrum
- arr_7: line depth for each detected line, from reference spectrum
- arr_8: line formation temperature for each detected line (Al Moulla et. al 2022)
- arr_9: S-index calculated from Ca H/K for each file
- arr_10: line depth of Mn I 5394.7 Å for each file
- arr_11: pearson correlation coefficient for each line rv with a daily linear trend (line diagnostic)
- arr_12: percent of files utilizing a given line (line diagnostic)
- arr_13: number of lines used for each file

## Other files:

`"postprocess.ipynb"`: Jupiter notebook for post processing outputs