import os
import pathlib
import numpy as np
import pandas as pd
import argparse as ap
from astropy.io import fits
from astropy.time import Time
from astropy.timeseries import LombScargle
from numpy.core.multiarray import interp
from scipy import stats
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_filter1d
from scipy.stats import pearsonr
from tqdm import tqdm
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

'''
get data from single fits file
'''
def fetch_single(path):
        # define the fits extensions
        fib = 'SCI' # SCIENCE fiber
        fits_extension_spectrum = fib + 'FLUX' # this is how we catalog the different bits of NEID data
        fits_extension_variance = fib + 'VAR' # variance (noise) extension of fits file
        fits_extension_wavelength = fib + 'WAVE' # wavelength solution extension
        fits_extension_blaze = fib + 'BLAZE' # wavelength solution extension
        # primary header
        hdul = fits.open(path)
        header = hdul[0].header
        # global properties of the spectrum
        berv  = header['SSBRV052'] * 1000 # m/s
        #drift = header['DRIFTRV0'] # m/s
        # the actual spectrum (in flux units)
        data_spec = hdul[1].data[17:-13,:]
        # the actual blaze (in flux units)
        blz_spec = hdul[15].data[17:-13,:]
        # the variance of the spectrum
        # var = fits.getdata(path, fits_extension_variance)[9:-5,:]
        var = np.ones_like(data_spec)
        var[data_spec>0] = np.sqrt(data_spec[data_spec>0])
        # the wavelength solution of the spectrum (natively in Angstroms)
        wsol = hdul[7].data[17:-13,:] # A
        # Shift to heliocentric frame, compensate for zero point offset
        wsol = wsol*(1 + ((berv-83285) / 299792458))
        # manually filter bad columns
        data_spec[:,434:451]   = 0
        data_spec[:,1930:1945] = 0
        # filter for orders with duplicates in x
        wsol      = list(wsol)[:109]      + list(wsol)[111:]
        data_spec = list(data_spec)[:109] + list(data_spec)[111:]
        var       = list(var)[:109]       + list(var)[111:]
        blz_spec  = list(blz_spec)[:109]  + list(blz_spec)[111:]
        # filter for nan (by creating pseudo orders)
        X = []; Y = []; E = []
        for x,y,e,b in zip(wsol,data_spec,var,blz_spec):
                b[~np.isfinite(y)] = 1
                b[~np.isfinite(b)] = 1
                b[b==0] = 1
                y[~np.isfinite(y)] = np.nan
                y[y<=0] = np.nan
                y[~np.isfinite(b)] = np.nan
                e[~np.isfinite(e)] = np.nan
                e[e<0] = np.nan
                X.append( x ); Y.append( y/b ); E.append(e/b )

        # convert to numpy arrays
        X = np.array(X)
        Y = np.array(Y)
        E = np.array(E)
        return X, Y, E

'''
creates reference spectrum from files in path, and a wavelength reference
'''
def create_ref_spectrum(path, reference):
        
        # directory for all files
        files = list(pathlib.Path(path).glob('*.fits'))

        # get reference file
        waveref,flux,error = fetch_single(reference)

        # initialize array for reference spectrum
        refspectrum = np.zeros(np.shape(waveref))

        # integrate fluxes for all files after interpolating them to the same wavelength array
        for k in tqdm(range(len(files)), desc="integrating reference"):

                # get file
                wavelength,flux,error = fetch_single(files[k])

                # normalize flux
                for i in range(len(flux)):
                        flux[i] = flux[i]/maximum_filter1d(np.where(np.isnan(flux[i]),-np.inf, flux[i]), size=1000)
                        
                # interpolate and integrate
                refspectrum += np.concatenate([interp(waveref[[j]],wavelength[j],flux[j]) for j in range(np.shape(flux)[0])])

        return waveref, refspectrum/len(files)

'''
load reference spectrum, create telluric mask and line windows
'''
def load_ref_spectrum(path, telpath, wvlpath, maskstrength):

        # Open telluric template
        y = fits.open(telpath)[0].data        
        x = fits.open(wvlpath)[0].data * 10 # angstrom

        # smooth out a bit to get rid of continuum
        y = y/maximum_filter1d(y, size=2)
        mask = np.where((np.abs(y-1) > (10**(-maskstrength))))[0]

        # create groups
        start = x[mask[np.where(np.diff(mask) != 1)]]
        end = x[mask[np.where(np.diff(mask) != 1)[0]+1]]
        start = np.insert(start, len(start), x[mask[-1]])
        end = np.insert(end, 0, x[mask[0]])   
       
        # Load reference spectrum
        result = np.load(path)
        wavelength = result["arr_0"]
        flux = result["arr_1"]

        # create telluric mask using grouups
        big_mask = (wavelength<0)
        for i in tqdm(range(len(start)), desc="constructing telluric mask"):
                big_mask |= ((wavelength>(start[i]-0.4))&(wavelength<(end[i]+0.4)))

        # Open solar template to find modeled lines
        template = pd.read_csv("T1o2_spec-2.csv")
        # shift from vacuum to air
        wave_w = template["wave"]*(1 - (83285/299792458))
        flux_w = template["flux"]
        temp_w = template["T1o2"]
        tempmin = wave_w[find_peaks(np.nanmax(flux_w)-flux_w, distance=5,height=.1, prominence=.1)[0]]

        '''
        Get local minima for absorption lines, get cubic spline model,       
        get local maxima to set windows for each line 
        '''
        minindices = []
        maxima = []
        csplines = []
        for i in range(np.shape(flux)[0]):
                csplines.append(CubicSpline(wavelength[i][np.isfinite(flux[i])], flux[i][np.isfinite(flux[i])]))
                maxima.append(find_peaks(flux[i], distance=1,height=.02, prominence=.02)[0])
                minlist = find_peaks(np.nanmax(flux[i])-flux[i], distance=5,height=.1, prominence=.1)[0]
                minindices.append(minlist[np.isin(minlist, np.where(big_mask[i] == False)[0])])

        minima = []
        contdiff = []
        linedepth = []
        templatemask = []
        temperatures = []
##        masscenter = []
##        smallwindow = []
##        jerkdistance = []
##        bisector = []
##        num_maxima = []
        boxlist = []

        # create line windows
        for i in tqdm(range(len(minindices)),desc="creating line windows"):
                
                #, massCenterOrd, swOrd, jDOrd, bisectorOrd, numMaximaOrd,
                min_ord, contdiff_ord, linedepth_ord, boxord, templateord, temperatureord = np.zeros(len(minindices[i])),np.zeros(len(minindices[i])), np.zeros(len(minindices[i])),\
                                                                                            np.zeros(len(minindices[i])),np.zeros(len(minindices[i])),np.zeros(len(minindices[i]))
                                   
                for j in range(len(minindices[i])):

                        # Get location of line peak
                        linemin = wavelength[i][minindices[i][j]]
                        if (np.min(np.abs(tempmin - linemin)) < 0.1):
                                templateord[j] = True
                        else:
                                templateord[j] = False

                        # get local maxima surrounding line
                        nearestmaxindex = np.argmin(np.abs(maxima[i] - minindices[i][j]))
                        if (wavelength[i][maxima[i][nearestmaxindex]] >  linemin) & (nearestmaxindex != 0):
                            othermax_ord = flux[i][maxima[i][nearestmaxindex - 1]]
                        elif (wavelength[i][maxima[i][nearestmaxindex]] <  linemin) & (nearestmaxindex != (len(maxima[i])-1)):
                            othermax_ord = flux[i][maxima[i][nearestmaxindex + 1]]
                        else:
                            othermax_ord = np.nan

                        # difference between maxima
                        contdiff_ord[j] = np.abs(othermax_ord - flux[i][maxima[i][nearestmaxindex]])

                        # box around window
                        box = np.abs(wavelength[i][maxima[i][nearestmaxindex]] - linemin)
                        boxord[j] = box

                        # avg temperature
                        temperatureord[j] = np.mean(temp_w[(wave_w > (linemin-box)) & (wave_w < (linemin+box))])

                        # line depth
                        linedepth_ord[j] = 1 - 2*flux[i][minindices[i][j]]/(othermax_ord+flux[i][maxima[i][nearestmaxindex]])

                        min_ord[j] = linemin

                minima.append(min_ord)
                contdiff.append(contdiff_ord)
                boxlist.append(boxord)
                linedepth.append(linedepth_ord)
                templatemask.append(templateord)
                temperatures.append(temperatureord)
                
        return wavelength, csplines, minima, maxima, contdiff, linedepth, boxlist, templatemask, temperatures

'''
Process RV for a file
'''
class FileRV(object):
        def __init__(self, params):
                self.params = params
        def __call__(self, file):
                csplines, minima, maxima, contdiff, linedepth, boxlist, templatemask = self.params
                # Currently working on single file, looping all files later
                wS, fS, eS = fetch_single(file)

                # Interpolate flux of reference to file
                flux = np.vstack([csplines[i](wS[i]) for i in range(np.shape(wS)[0])])

                try:
                        # get Ca II H/K wavelengths
                        CaH = 3968.47
                        CaK = 3933.66

                        # Calculate s-index
                        CaHflux1 = np.nansum(fS[10][(wS[10] < (CaH+0.545)) & (wS[10] > (CaH-0.545))])
                        CaHflux2 = np.nansum(fS[9][(wS[9] < (CaH+0.545)) & (wS[9] > (CaH-0.545))])
                        CaKflux1 = np.nansum(fS[8][(wS[8] < (CaK+0.545)) & (wS[8] > (CaK-0.545))])
                        CaKflux2 = np.nansum(fS[9][(wS[9] < (CaK+0.545)) & (wS[9] > (CaK-0.545))])
                        flux3900 = (CaHflux2/CaHflux1)*np.nansum(fS[8][(wS[8] < (3910)) & (wS[8] > (3890))])
                        flux4000 =(CaKflux2/CaKflux1)* np.nansum(fS[10][(wS[10] < (4010)) & (wS[10] > (3990))])
                        s_index = (CaHflux2+CaKflux2)/(flux3900+flux4000)
                        #RHKprime = pyasl.SMW_RHK(ccfs='noyes', afc='middelkoop', rphot='noyes').SMWtoRHK(s_index, 5778, 0.656, lc='ms', verbose=False)[0]

                        # Get Mn I wavelengths
                        Mn1 = np.where((minima[50] > 5394) & (minima[50] < 5396))[0]
                        Mn2 = np.where((minima[51] > 5394) & (minima[51] < 5396))[0]
                except:
                        raise ValueError('masking obscures critical lines')

                # Calculate rv, rv errors, and other parameters
                rv = []
                rverror = []
                corrcoeff = []
                linewidth = []
                mn_linedepth = 0

                # calculate rvs for each order
                for i in range(len(minima)):

                        # continuum division
                        fluxcont = fS[i]/maximum_filter1d(np.where(np.isnan(fS[i]),-np.inf, fS[i]), size=1000)
                        errorCont = eS[i]/maximum_filter1d(np.where(np.isnan(fS[i]),-np.inf, fS[i]), size=1000)
                        # Initialize arrays
                        rvord, rverror_ord, corrcoeff_ord, linewidth_ord = np.zeros(len(minima[i])),np.zeros(len(minima[i])),np.zeros(len(minima[i])),np.zeros(len(minima[i]))
                        # Measure line depth of Mn I 5394.47
                        if ((i==50) | (i==51)):
                                mnbox = np.abs(wS[i][maxima[i]][np.argmin(np.abs(wS[i][maxima[i]] - 5394.7))] - 5394.7)
                                relflux = fluxcont[(wS[i] > (5394.7-mnbox)) & (wS[i] < (5394.7+mnbox))]
                                mn_linedepth += 1 - (np.min(relflux))/np.max(relflux)

                        # iterate over lines in each order
                        for j in range(len(minima[i])):

                                # get line windows, line minimum, indices, and line flux
                                box = boxlist[i][j]
                                linemin = minima[i][j]
                                indices = np.where((wS[i] < (box+linemin)) & (wS[i] > (linemin-box)))
                                fluxspec = fluxcont[indices]
                                # minimum pixel length of window
                                if ((len(indices[0]) > 10) & (len(indices[0]) < 100) & (~np.isnan(linemin)) & (~np.isnan(fluxspec).any()) & (linedepth[i][j] > 0.005) & (contdiff[i][j] < 0.05)
                                    & (templatemask[i][j] == True)):
                                        # get wavelength of interpolated target spectrum in the window
                                        waveline = wS[i][indices]
                                        der = csplines[i](waveline, 1)
                                        halfmax = 0.5*(np.min(fluxspec)+np.max(fluxspec))
                                        linewidth_ord[j] = np.abs(waveline[waveline < linemin][np.argmin(np.abs(fluxspec[waveline < linemin] - halfmax))] -
                                                        waveline[waveline > linemin][np.argmin(np.abs(fluxspec[waveline > linemin] - halfmax))])
                                        # try/except linear regression
                                        location = ~np.isnan(flux[i][indices]) & ~np.isnan(fluxspec)
                                        # linear least-squares regression
                                        try:
                                                def model(S,A,Adl):
                                                        return A * S + Adl*der[location]
                                                bestpars,parscov = curve_fit(model,flux[i][indices][location],
                                                                        fluxspec[location], sigma = errorCont[indices][location], absolute_sigma=True)
        ##                                        covInv = np.linalg.inv(np.diag(errorCont[indices][location]**2))
        ##                                        designMatrix = np.vstack((flux[i][indices][location], csplines[i](waveline[location], 1)))
        ##                                        parscov = np.linalg.inv(designMatrix @ covInv @ designMatrix.T)
        ##                                        bestpars = parscov @ designMatrix @ covInv @ fluxspec[location]
                                                corrcoeff_ord[j] = np.sqrt(np.square(parscov[1][0])/(parscov[1][1]*parscov[0][0]))
                                                rvord[j] = (-299792458*bestpars[1]/(bestpars[0]*linemin))
                                                rverror_ord[j] = (299792458/linemin)*np.sqrt((parscov[1][1]/bestpars[0]**2)\
                                                                                        + (np.sqrt(parscov[0][0])*bestpars[1]/(bestpars[0]**2))**2)
                                        except:
                                                #print(np.min(np.abs(maximum_filter1d(fS[i], size=2000))))
                                                raise ValueError('linear regression failed')

                                else:
                                        rvord[j] = np.nan
                                        rverror_ord[j] = np.nan
                                        corrcoeff_ord[j] = np.nan
                                        linewidth_ord[j] = np.nan
                        rv = np.concatenate((rv, rvord))
                        rverror = np.concatenate((rverror, rverror_ord))
                        corrcoeff = np.concatenate((corrcoeff, corrcoeff_ord))
                        linewidth = np.concatenate((linewidth, linewidth_ord))

                np.savez('npz/'+ file.name[:-5]+"_rv", rv, rverror, corrcoeff, linewidth)
                #print("Processed", file)
                return s_index, mn_linedepth*0.5

if __name__ == "__main__":

        parser = ap.ArgumentParser(prog="Process spectra",description="")

        parser.add_argument('filedir', help="file directory for target files")
        parser.add_argument('-c', '--cpucount', help="specify number of cpus to use")
        parser.add_argument('-t', '--telluricmask',
                            help="specify telluric mask strength as a fraction of line depth (default = 1e-4)")
        parser.add_argument('-i', '--noint', action='store_true',
                            help="disable automatic creation of reference spectrum if one already exists")

        args = parser.parse_args()

        if args.telluricmask == None:
                print("defaulting to telluric mask strength of 1e-4")
                args.telluricmask = 4
                
        # directory for all files
        files = list(pathlib.Path(str(args.filedir)).glob('*.fits'))
        
        if not args.noint:

                waveref, refspectrum = create_ref_spectrum(str(args.filedir), files[0])
                # Save reference spectrum
                np.savez("refspectrum.npz", waveref, refspectrum)

        waveref, csplines, minima, maxima, contdiff, linedepth, boxlist, templatemask, temperatures =\
                 load_ref_spectrum("refspectrum.npz",'TAPAS_WMKO_NORAYLEIGH_SPEC.fits', 'TAPAS_WMKO_NORAYLEIGH_SPEC_WVL.fits', int(args.telluricmask))

        # Create directory for npz output files
        if not os.path.exists('npz'):
                os.makedirs('npz')

        # Parallel process files with args
        if args.cpucount == None:
                print("using default Pool")
                pool = Pool()
        else:
                pool = Pool(int(args.cpucount))
                        
        output = np.asarray(pool.map(FileRV((csplines, minima, maxima, contdiff, linedepth, boxlist, templatemask)), files))
                              
        # flatten line minima array
        wavelines = np.concatenate(minima)

        # Open npz directory
        files = list(pathlib.Path('npz').glob('*.npz'))

        avg_rverr_line = np.empty(shape=(len(files), len(wavelines)))

        # Get average rv error per line
        for i in range(len(files)):
                arrays = np.load(files[i])
                avg_rverr_line[i] = arrays["arr_1"]
        avg_rverr_line = np.nanmean(avg_rverr_line, axis=0)

        # Create mask for duplicates
        dupmask = np.zeros(len(wavelines), dtype=bool)
        for i in range(len(wavelines)):
                # Calculate difference between a line and the entire list to locate duplicates, within a 0.1 angstrom tolerance
                zL = np.abs(wavelines - wavelines[i])
                duplicates = np.where((zL < 0.1))[0]
                if (len(duplicates) > 1):
                        dupmask[duplicates[duplicates != duplicates[np.argmin(avg_rverr_line[duplicates])]]] = True
        dupmask[np.where(avg_rverr_line == np.nan)] = True

        # Flatten arrays, remove duplicates
        wavelines = wavelines[~dupmask]
        linedepth = np.concatenate(linedepth)[~dupmask]
        contdiff = np.concatenate(contdiff)[~dupmask]
        temperatures = np.concatenate(temperatures)[~dupmask]

        # initialize other arrays
        meansr,means,error,neidrv,time,angle,numlines=np.zeros(len(files)),np.zeros(len(files)),np.zeros(len(files)),np.zeros(len(files)),np.zeros(len(files)),np.zeros(len(files)),np.zeros(len(files))

        rvarrays = np.empty(shape=(len(files), len(wavelines)))
        rverr_arrays = np.empty(shape=(len(files), len(wavelines)))
        line_search = np.empty(shape=(len(files), len(wavelines)))

        # Get high trend RVs in IR and neidrv, time, and solar altitude
        for i in tqdm(range(len(files)), desc="pearson correlation"):

                arrays = np.load(files[i])

                hdul = fits.open(str(args.filedir)+files[i].name[:-7]+".fits")

                angle[i] = hdul[0].header['SUNAGL']
                neidrv[i] = hdul[12].header['CCFRVMOD']*1000
                time[i] = hdul[12].header['CCFJDMOD']

                rv = arrays["arr_0"][~dupmask]
                rverr = arrays["arr_1"][~dupmask]

                rvred = rv[wavelines > 7000]
                rverr_red = rverr[wavelines > 7000]

                cutr = np.where((rverr_red < 3*np.nanmean(rverr_red)) &
                        (np.abs(rvred - np.nanmean(rvred)) < 3*np.nanstd(rvred)))
                meansr[i] = np.sum(rvred[cutr]/(rverr_red[cutr]**2))/np.sum(1/(rverr_red[cutr]**2))

                line_search[i][(rverr < 3*np.nanmean(rverr)) & (np.abs(rv - np.nanmean(rv)) < 3*np.nanstd(rv))] = 1
                
                rvarrays[i] = rv
                rverr_arrays[i] = rverr

        # Calculate corr coeff for each line with respect to high trend lines for filtering
        pearsoncorr, perline = np.zeros(len(rvarrays.T)),np.zeros(len(rvarrays.T))

        for i in range(len(rvarrays.T)):
                line = rvarrays.T[i]
                filelist = line_search[:,i]
                perline[i] = len(filelist[filelist == 1])/len(filelist)
                try:
                        pearsoncorr[i] = pearsonr(meansr[~np.isnan(line)], line[~np.isnan(line)])[0]
                except:
                        pearsoncorr[i] = np.nan

        # Calculate bulk rv
        for i in range(len(files)):
                rv = rvarrays[i]
                rverr = rverr_arrays[i]

                cut = np.where((rverr < 3*np.nanmean(rverr)) &
                (np.abs(rv - np.nanmean(rv)) < 3*np.nanstd(rv))& (np.abs(pearsoncorr) < 0.5) & (perline > 0.01))

                numlines[i] = len(rv[cut])

                means[i] = np.sum(rv[cut]/(rverr[cut]**2))/np.sum(1/(rverr[cut]**2))
                error[i] = np.mean(rverr[cut])

        # Output rvs and calculated parameters
        np.savez("all_lines", rvarrays, rverr_arrays)
        np.savez("output_file", means, error, neidrv, time, angle, wavelines, contdiff, linedepth, temperatures, output[:,0], output[:,1], pearsoncorr, perline, numlines)
