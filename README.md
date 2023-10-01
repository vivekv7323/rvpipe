# rvpipe
## A command-line pipeline for fast line-by-line rv processing of NEID solar spectra.

## Required files: 
- Line temperature table (`"T1o2_spec-2.csv"`)
- Telluric line template (`"TAPAS_WMKO_NORAYLEIGH_SPEC.fits"`, `"TAPAS_WMKO_NORAYLEIGH_WVL.fits"`)

## Arguments:

`python3 rvpipe.py ["target file directory"] [-c] [cpu count] [-t] [telluric mask strength] [-i]`

`["target file directory"]`: Enter directory to target files for processing.

`[-c] [cpu count]`: Specify number of cpus to use (default is all available)

`[-t] [telluric mask strength]`: Specify telluric mask strength as a fraction of line depth (default = 1e-4)

`[-i]`: Disable automatic creation of reference spectrum (`"refspectrum.npz"`) if one already exists from a previous run

## Outputs:

`"all_lines.npz"`: Contains two numpy arrays, `arr_0` being an array of rvs for each file and line in the format `[file, line]`, `arr_1` being the corresponding errors
`"output_file.npz"`: Contains 14 numpy arrays.
- arr_0: mean RV for each file
- arr_1: RV error for each file
- arr_2: mean RV from the NEID pipeline for each file
- arr_3: time (Julian Date)
- arr_4: solar altitude (deg)
- arr_5: wavelength (Angstroms) for each detected line
- arr_6: continuum difference for each detected line
- arr_7: line depth for each detected line
- arr_8: line formation temperature for each detected line (Al Moulla et. al 2022)
- arr_9: S-index calculated from Ca H/K for each file
- arr_10: line depth of Mn I 5394.7 A for each file
- arr_11: pearson correlation coefficient for each line rv with a daily linear trend (line diagnostic)
- arr_12: Percent of files utilizing a given line (line diagnostic)
- arr_13: Number of lines used for each file

## Other files:

`"postprocess.ipynb"`: Jupiter notebook for post processing Outputs