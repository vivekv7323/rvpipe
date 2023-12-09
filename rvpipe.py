'''
Author: Vivek Vijayakumar

Imports:
'''
import os
import pathlib
import numpy as np
import pandas as pd
import argparse as ap
from astropy.io import fits
from astropy.time import Time
from astropy.timeseries import LombScargle
from numpy.core.multiarray import interp
import numpy.lib.recfunctions as rf
from scipy import stats
from scipy.signal import argrelextrema
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
def get_data(path, cropL, cropR, filetype):
        
        # distinguish between filetype (neid, harpn)
        if (filetype == "neid"):
                
                # primary header
                hdul = fits.open(path)
                header = hdul[0].header
                
                # global properties of the spectrum
                berv  = header['SSBRV052'] * 1000 # m/s
                
                #drift = header['DRIFTRV0'] # m/s
                
                # the actual spectrum (in flux units)
                data_spec = hdul[1].data[17:-13,:]
                
                # manually filter bad columns
                data_spec[:,434:451]   = 0
                data_spec[:,1930:1945] = 0
                data_spec = data_spec[:, cropL:cropR]
                
                # the actual blaze (in flux units)
                blz_spec = hdul[15].data[17:-13,cropL:cropR]
                
                # the standard deviation of the spectrum from variance
                #var = np.ones_like(data_spec)
                #var[data_spec>0] = np.sqrt(data_spec[data_spec>0])
                var = np.sqrt(hdul[4].data[17:-13,:])
                var[:,434:451]   = 0
                var[:,1930:1945] = 0
                var = var[:, cropL:cropR]
                
                # the wavelength solution of the spectrum (natively in Angstroms)
                wsol = hdul[7].data[17:-13,cropL:cropR] # A
                
                # Shift to heliocentric frame, compensate for zero point offset
                wsol = wsol*(1 + ((berv-83285) / 299792458))
                
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
                
                # get bulk rv, time, and altitude
                measurements = [hdul[12].header['CCFRVMOD']*1000,hdul[12].header['CCFJDMOD'],hdul[0].header['SUNAGL']]
                
        elif(filetype == "harpn"):
                
                # primary header
                hdul = fits.open(path)
                hdul1 = np.asarray(rf.structured_to_unstructured(hdul[1].data))
                
                # harpn 1D is already clean, can proceed to getting wavelength solution, flux, and error
                X, Y, E = np.zeros((1, np.shape(hdul1)[0])),np.zeros((1, np.shape(hdul1)[0])),np.zeros((1, np.shape(hdul1)[0]))
                
                # using air wavelength
                X[0,:] = hdul1[:, 1]
                Y[0,:] = hdul1[:, 2]
                E[0,:] = hdul1[:, 3]
                
                # get bulk rv, time, and altitude
                measurements = [hdul[0].header["HIERARCH TNG QC CCF RV"]*1000, hdul[0].header["HIERARCH TNG QC BJD"],hdul[0].header["EL"]]

        elif (filetype == "adp"):
                
                hdul = fits.open(path)

                length = len(hdul[1].data[0][0])

                X, Y, E = np.zeros((1, length)),np.zeros((1, length)),np.zeros((1, length))

                X[0,:],Y[0,:] = hdul[1].data[0][0], hdul[1].data[0][1]

                Y[Y==0] = np.nan
                E = np.sqrt(Y)

                try:
                        adprv = hdul[0].header["HIERARCH ESO QC CCF RV"]
                except:
                        adprv = np.nan
                try:
                        adptime = hdul[0].header["HIERARCH ESO DRS BJD"]
                except:
                        adptime = hdul[0].header["HIERARCH ESO QC BJD"]
                try:
                        try:
                                adpalt = hdul[0].header["HIERARCH ESO TEL ALT"]
                        except:
                                adpalt = hdul[0].header["HIERARCH ESO TEL1 ALT"]
                except:
                        adpalt = np.nan

                measurements = [adprv, adptime, adpalt]
                
        else:
                raise ValueError("no filetype specified")
        
        return X, Y, E, measurements

'''
creates reference spectrum from files in path, and a wavelength reference
'''
class RefSpec(object):
        
        def __init__(self, params):
                self.params = params
                
        def __call__(self, file):

                waveref, cropL, cropR, filetype = self.params

                # get file
                wavelength,flux,error,measurements = get_data(file, cropL, cropR, filetype)

                # overall mean error
                err = np.nanmedian(error)/np.nanmean(flux)

                # normalize flux
                for i in range(len(flux)):
                        flux[i] = flux[i]/maximum_filter1d(np.where(np.isnan(flux[i]),-np.inf, flux[i]), size=1000)
                        
                # interpolate and integrate
                big2Darr = np.concatenate([interp(waveref[[j]],wavelength[j],flux[j]) for j in range(np.shape(flux)[0])])

                return big2Darr, err

class RefSpecBuf(object):
        
        def __init__(self, params):
                self.params = params
                
        def __call__(self, files):

                waveref, cropL, cropR, filetype = self.params

                # array for integrating files      
                buf_3D = np.zeros((len(files), np.shape(waveref)[0], np.shape(waveref)[1]))
                
                # error estimate
                err = np.zeros(len(files))

                for k in range(len(files)):
                        # get file
                        wavelength,flux,error,measurements = get_data(files[k], cropL, cropR, filetype)

                        # overall mean error
                        err[k] = np.nanmedian(error)/np.nanmean(flux)

                        # normalize flux
                        for i in range(len(flux)):
                                flux[i] = flux[i]/maximum_filter1d(np.where(np.isnan(flux[i]),-np.inf, flux[i]), size=1000)

                        buf_3D[k] = np.concatenate([interp(waveref[[j]],wavelength[j],flux[j]) for j in range(np.shape(flux)[0])])


                return np.nanmean(buf_3D, axis=0)*len(files), np.mean(err[err<1])

'''
load reference spectrum, create telluric mask and line windows
'''
def load_ref_spectrum(path, telpath, wvlpath, maskdepth, maskdev, argmask, minlinedepth):

        # Open telluric template
        y = fits.open(telpath)[0].data        
        x = fits.open(wvlpath)[0].data * 10 # angstrom

        # smooth out a bit to get rid of continuum
        y = y/maximum_filter1d(y, size=2)

        # create mask by selecting wavelengths where tellurics have a line depth exceeding maskdepth
        mask = np.where((np.abs(y-1) > (10**(-maskdepth))))[0]

        # create groups of wavelengths corresponding to selected telluric regions
        start = x[mask[np.where(np.diff(mask) != 1)]]
        end = x[mask[np.where(np.diff(mask) != 1)[0]+1]]
        start = np.insert(start, len(start), x[mask[-1]])
        end = np.insert(end, 0, x[mask[0]])   
       
        # Load reference spectrum
        result = np.load(path)
        wavelength = result["arr_0"]
        flux = result["arr_1"]
        avgerr = 0.5*10**np.round(np.log10(result["arr_2"]))

        # create telluric mask using grouups
        big_mask = (wavelength<0)
        for i in tqdm(range(len(start)), desc="constructing telluric mask"):
                big_mask |= ((wavelength>(start[i]-maskdev))&(wavelength<(end[i]+maskdev)))

        # Open solar template to find modeled lines
        template = pd.read_csv("T1o2_spec-2.csv")
        
        # shift from vacuum to air
        wave_w = template["wave"]*(1 - (83285/299792458))
        flux_w = template["flux"]
        temp_w = template["T1o2"]
        tempmin = wave_w[find_peaks(np.nanmax(flux_w)-flux_w, distance=5,height=0, prominence=.01)[0]]

        '''
        Get local minima for absorption lines, get cubic spline model,       
        get local maxima to set windows for each line 
        '''
        minindices = []
        maxima = []
        csplines = []
        
        # Create cubic spline model, obtain maxima and minima in spectra (except in telluric filtered regions)
        for i in range(np.shape(flux)[0]):
                csplines.append(CubicSpline(wavelength[i][np.isfinite(flux[i])], flux[i][np.isfinite(flux[i])]))
                maxima.append(find_peaks(flux[i], distance=1,height=0, prominence=avgerr)[0])
                if (minlinedepth > 5*avgerr):
                        minlist = find_peaks(np.nanmax(flux[i])-flux[i], distance=5,height=0, prominence=minlinedepth)[0]
                else:
                        minlist = find_peaks(np.nanmax(flux[i])-flux[i], distance=5,height=0, prominence=5*avgerr)[0]
                minindices.append(minlist[np.isin(minlist, np.where(big_mask[i] == False)[0])])

        minima = []
        linedepth = []
        contdiff = []
        contavg = []
        masscenter = []
        jerkdistance = []
        bisectormax = []
        templatemask = []
        temperatures = []
        boxlist = []

        # create line windows
        for i in tqdm(range(len(minindices)),desc="creating line windows"):
                
                min_ord, linedepth_ord, contdiff_ord, contavg_ord, mass_ord, jd_ord, bisector_ord, templateord, temperatureord, boxord =\
                        np.zeros(len(minindices[i])),np.zeros(len(minindices[i])), np.zeros(len(minindices[i])),\
                                np.zeros(len(minindices[i])),np.zeros(len(minindices[i])),np.zeros(len(minindices[i])),\
                                np.zeros(len(minindices[i])),np.zeros(len(minindices[i])),np.zeros(len(minindices[i])),np.zeros(len(minindices[i]))
                                   
                for j in range(len(minindices[i])):


                        # Get location of line peak
                        linemin = wavelength[i][minindices[i][j]]
                        if ((np.min(np.abs(tempmin - linemin)) < 0.1) or argmask):
                                templateord[j] = True
                        else:
                                templateord[j] = False

                        # get local maxima surrounding line
                        nearestmaxindex = np.argmin(np.abs(maxima[i] - minindices[i][j]))
                        if (wavelength[i][maxima[i][nearestmaxindex]] >  linemin) & (nearestmaxindex != 0):
                            othermax = flux[i][maxima[i][nearestmaxindex - 1]]
                        elif (wavelength[i][maxima[i][nearestmaxindex]] <  linemin) & (nearestmaxindex != (len(maxima[i])-1)):
                            othermax = flux[i][maxima[i][nearestmaxindex + 1]]
                        else:
                            othermax = np.nan

                        '''
                        Calculate all filtering parameters
                        '''

                        # line depth
                        ld = 1 - 2*flux[i][minindices[i][j]]/(othermax+flux[i][maxima[i][nearestmaxindex]])
                        linedepth_ord[j] = ld

                        # box around window
                        box = np.abs(wavelength[i][maxima[i][nearestmaxindex]] - linemin)
                        boxord[j] = box

                        # the spectral window
                        window = wavelength[i][(wavelength[i] < box+linemin) & (wavelength[i] > linemin-box)]
                        
                        # mass center
                        cflux = csplines[i](window)
                        der1 = csplines[i](window, 1)
                        der2 = csplines[i](window, 2)
                        dermax = window[der1 == np.max(der1)][0]
                        dermin = window[der1 == np.min(der1)][0]
                        mass_ord[j] = np.abs(((dermax+dermin)/2)-linemin)/(dermax-dermin)

                        # difference between maxima
                        contdiff_ord[j] = np.abs(othermax - flux[i][maxima[i][nearestmaxindex]])
                        
                        # average of maxima
                        contavg_ord[j] = (othermax + flux[i][maxima[i][nearestmaxindex]])/2

                        # calculate jerk distance (try/except is due to possibility of window of length 1)
                        try:
                                jd_ord[j] = np.abs(cflux[der2 == np.min(der2[~(der2 == np.min(der2))])] - cflux[der2 == np.min(der2)])[0]/ld

                        except:
                                jd_ord[j] = np.nan

                        # calculate bisector                                
                        left = np.flip(cflux[window < linemin])
                        right = cflux[window > linemin]
                        if (len(left) > len(right)):
                            bisector = (left[:len(right)]+right)/2
                        elif (len(left) < len(right)):
                            bisector  = (right[:len(left)]+left)/2
                        else:
                            bisector = (left+right)/2

                        bisector_ord[j] = len(argrelextrema(bisector, np.greater)[0])+len(argrelextrema(bisector, np.less)[0])

                        # avg temperature
                        temperatureord[j] = np.mean(temp_w[(wave_w > (linemin-box)) & (wave_w < (linemin+box))])

                        min_ord[j] = linemin

                minima.append(min_ord)
                linedepth.append(linedepth_ord)
                contdiff.append(contdiff_ord)
                contavg.append(contavg_ord)
                masscenter.append(mass_ord)
                jerkdistance.append(jd_ord)
                bisectormax.append(bisector_ord)
                templatemask.append(templateord)
                temperatures.append(temperatureord)
                boxlist.append(boxord)
                
        return wavelength, csplines, minima, maxima, linedepth, contdiff, contavg, masscenter, jerkdistance, bisectormax, templatemask, temperatures, boxlist

'''
Process RV for a file
'''
class FileRV(object):
        def __init__(self, params):
                self.params = params
        def __call__(self, file):
                csplines, minima, maxima, linedepth, contdiff, contavg, masscenter, jerkdistance,\
                        bisectormax, templatemask, boxlist, filterpars, cropL, cropR, filetype, path_intermed, filecompress = self.params
                
                wS, fS, eS, measurements = get_data(file, cropL, cropR, filetype)

                # Interpolate flux of reference to file
                flux = np.vstack([csplines[i](wS[i]) for i in range(np.shape(wS)[0])])

                if (filetype == "neid"):
                        ca_ind = 8
                        ca_ind2 = 9
                        ca_ind3 = 10
                        mn_ind = 50
                        mn_ind2 = 51
                elif ((filetype == "harpn") or (filetype == "adp")):
                        ca_ind = 0
                        ca_ind2 = 0
                        ca_ind3 = 0
                        mn_ind = 0
                        mn_ind2 = 0

                try:
                        # get Ca II H/K wavelengths
                        CaH = 3968.47
                        CaK = 3933.66

                        # Calculate s-index
                        CaHflux1 = np.nansum(fS[ca_ind3][(wS[ca_ind3] < (CaH+1)) & (wS[ca_ind3] > (CaH-1))])
                        CaHflux2 = np.nansum(fS[ca_ind2][(wS[ca_ind2] < (CaH+1)) & (wS[ca_ind2] > (CaH-1))])
                        CaKflux1 = np.nansum(fS[ca_ind][(wS[ca_ind] < (CaK+1)) & (wS[ca_ind] > (CaK-1))])
                        CaKflux2 = np.nansum(fS[ca_ind2][(wS[ca_ind2] < (CaK+1)) & (wS[ca_ind2] > (CaK-1))])
                        flux3900 = (CaHflux2/CaHflux1)*np.nansum(fS[ca_ind][(wS[ca_ind] < (3910)) & (wS[ca_ind] > (3890))])
                        flux4000 =(CaKflux2/CaKflux1)* np.nansum(fS[ca_ind3][(wS[ca_ind3] < (4010)) & (wS[ca_ind3] > (3990))])
                        s_index = (CaHflux2+CaKflux2)/(flux3900+flux4000)
                        #RHKprime = pyasl.SMW_RHK(ccfs='noyes', afc='middelkoop', rphot='noyes').SMWtoRHK(s_index, 5778, 0.656, lc='ms', verbose=False)[0]

                        # Get Mn I wavelengths
                        Mn1 = np.where((minima[mn_ind] > 5394) & (minima[mn_ind] < 5396))[0]
                        Mn2 = np.where((minima[mn_ind2] > 5394) & (minima[mn_ind2] < 5396))[0]
                except:
                        raise ValueError('masking obscures critical lines')

                # Calculate rv, rv errors, and other parameters
                rv = []
                rverror = []
                corrcoeff = []
                linewidth = []
                pixelindices = []
                linedepthline = []
                mn_linedepth = 0

                # calculate rvs for each order
                for i in range(len(minima)):

                        # continuum division
                        fluxcont = fS[i]/maximum_filter1d(np.where(np.isnan(fS[i]),-np.inf, fS[i]), size=1000)
                        errorCont = eS[i]/maximum_filter1d(np.where(np.isnan(fS[i]),-np.inf, fS[i]), size=1000)
                        # Initialize arrays
                        rvord, rverror_ord, corrcoeff_ord, linewidth_ord, pixelindices_ord, linedepth_ord = np.zeros(len(minima[i])),\
                                np.zeros(len(minima[i])),np.zeros(len(minima[i])),np.zeros(len(minima[i])),np.zeros(len(minima[i])),np.zeros(len(minima[i]))
                        # Measure line depth of Mn I 5394.47
                        if ((i==mn_ind) | (i==mn_ind2)):
                                mnbox = np.abs(wS[i][maxima[i]][np.argmin(np.abs(wS[i][maxima[i]] - 5394.7))] - 5394.7)
                                relflux = fluxcont[(wS[i] > (5394.7-mnbox)) & (wS[i] < (5394.7+mnbox))]
                                try:
                                        mn_linedepth += 1 - (np.min(relflux))/np.max(relflux)
                                except:
                                        mn_linedepth = np.nan

                        # iterate over lines in each order
                        for j in range(len(minima[i])):

                                # get line windows, line minimum, indices, and line flux
                                box = boxlist[i][j]
                                linemin = minima[i][j]
                                indices = np.where((wS[i] < (box+linemin)) & (wS[i] > (linemin-box)))
                                pixelindices_ord[j] = len(indices[0])
                                fluxspec = fluxcont[indices]
                                # minimum pixel length of window
                                if ((len(indices[0]) > filterpars[0]) & (len(indices[0]) < filterpars[1]) & (~np.isnan(linemin)) &
                                    (~np.isnan(fluxspec).any()) & (linedepth[i][j] > filterpars[2]) & (contdiff[i][j] < filterpars[3]) &
                                    (templatemask[i][j] == True) & (contavg[i][j] > filterpars[4]) & (masscenter[i][j] < filterpars[5]) &
                                    (jerkdistance[i][j] < filterpars[6]) & (bisectormax[i][j] < filterpars[7])):
                                        # get wavelength of interpolated target spectrum in the window
                                        waveline = wS[i][indices]
                                        der = csplines[i](waveline, 1)
                                        halfmax = 0.5*(np.min(fluxspec)+np.max(fluxspec))
                                        linewidth_ord[j] = np.abs(waveline[waveline < linemin][np.argmin(np.abs(fluxspec[waveline < linemin] - halfmax))] -
                                                        waveline[waveline > linemin][np.argmin(np.abs(fluxspec[waveline > linemin] - halfmax))])
                                        linedepth_ord[j] = 1 - 2*np.min(fluxspec)/(fluxspec[0]+fluxspec[-1])
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
                                        pixelindices_ord[j] = np.nan
                                        linedepth_ord[j] = np.nan
                                        
                        rv = np.concatenate((rv, rvord))
                        rverror = np.concatenate((rverror, rverror_ord))
                        corrcoeff = np.concatenate((corrcoeff, corrcoeff_ord))
                        linewidth = np.concatenate((linewidth, linewidth_ord))
                        pixelindices = np.concatenate((pixelindices, pixelindices_ord))
                        linedepthline = np.concatenate((linedepthline, linedepth_ord))

                if filecompress:
                        np.savez_compressed(os.path.join(path_intermed, file.name[:-5]+"_rv"), rv, rverror, corrcoeff, linewidth, pixelindices,\
                                 linedepthline,[s_index, mn_linedepth*0.5, measurements[0], measurements[1], measurements[2]])
                else:
                        np.savez(os.path.join(path_intermed, file.name[:-5]+"_rv"), rv, rverror, corrcoeff, linewidth, pixelindices,\
                                 linedepthline,[s_index, mn_linedepth*0.5, measurements[0], measurements[1], measurements[2]])                        

if __name__ == "__main__":

        parser = ap.ArgumentParser(prog="Process spectra",description="")

        parser.add_argument('filedir', help="file directory for target files")
        parser.add_argument('filetype', help="type of data file (NEID, HARPN, etc.)")

        parser.add_argument('-imf', '--intermedfiles',
                            help="file directory for intermediate files")

        parser.add_argument('-c', '--cpucount', help="specify number of cpus to use")
        parser.add_argument('-buf', '--buffer',
                    help="specify number of buffers for buffered integration")
        
        parser.add_argument('-td', '--telluricmaskdepth',
                            help="specify telluric mask strength as related to telluric line depth (default = 4, corresponding to a line depth of 1e-4)")
        parser.add_argument('-tdv', '--telluricmaskdev',
                            help="specify cut off distance from tellurics in angstroms (default = 0.05 Å)")
                            
##        parser.add_argument('-', '--minwavelength',
##                            help="specify minimum wavelength to use for analysis in angstroms")
##        parser.add_argument('-', '--maxwavelength',
##                            help="specify maximum wavelength to use for analysis in angstroms")
        
        parser.add_argument('-mlw', '--minlinewidth',
                            help="specify minimum line width in pixels")
        parser.add_argument('-xlw', '--maxlinewidth',
                            help="specify maximum line width in pixels")
        parser.add_argument('-mld', '--minlinedepth',
                            help="specify minimum line depth")
        parser.add_argument('-mcd', '--maxcontdiff',
                            help="specify maximum continuum difference")
        parser.add_argument('-mca', '--mincontavg',
                            help="specify minimum continuum average")
        parser.add_argument('-mc', '--masscenter',
                            help="specify maximum mass center")
        parser.add_argument('-jd', '--jerkdistance',
                            help="specify maximum jerk distance")
        parser.add_argument('-b', '--bisector',
                            help="specify maximum number of bisector extrema")
        
        parser.add_argument('-tm', '--templatemask', action='store_false',
                            help="restrict lines to those in the line template")
        parser.add_argument('-ni', '--noint', action='store_false',
                            help="disable automatic creation of reference spectrum if one already exists")
        parser.add_argument('-np', '--noproc', action='store_false',
                            help="disable file processing")
        parser.add_argument('-fc', '--filecompress', action='store_false',
                    help="compress intermediate files")
        


        args = parser.parse_args()

        # crop regions exclusively for NEID, no inclusion as a parameter yet
        cropL = 1500
        cropR = 7500

        # Parallel process files with args
        if args.cpucount == None:
                print("using default Pool")
                pool = Pool()
        else:
                pool = Pool(int(args.cpucount))

        if args.telluricmaskdepth == None:
                print("defaulting to telluric mask depth of 1e-4")
                args.telluricmaskdepth = 4
        if args.telluricmaskdev == None:
                print("defaulting to telluric cut-off of 0.1 Å")
                args.telluricmaskdev = 0.2
                
##        if args.minwavelength == None:
##                print("defaulting to no minimum wavelength")
##                args.minwavelength = 0
##        if args.maxwavelength == None:
##                print("defaulting to maximum wavelength of 7000 Å")
##                args.maxwavelength = 7000
                
        if args.minlinewidth == None:
                print("defaulting to minimum line width of 10 pixels")
                args.minlinewidth = 10
        if args.maxlinewidth == None:
                print("defaulting to maximum line width of 250 pixels")
                args.maxlinewidth = 250
        if args.minlinedepth == None:
                print("defaulting to minimum line depth of 0.01")
                args.minlinedepth = 0.05
        if args.maxcontdiff == None:
                print("defaulting to maximum continuum difference of 0.05")
                args.maxcontdiff = 0.05
        if args.mincontavg == None:
                print("defaulting to minimum continuum average of 0.8")
                args.mincontavg = 0.8
        if args.masscenter == None:
                print("defaulting to maximum mass center of 0.2")
                args.masscenter = 0.2
        if args.jerkdistance == None:
                print("defaulting to maximum jerk distance of 0.25")
                args.jerkdistance = 0.25
        if args.bisector == None:
                print("defaulting to maximum bisector extrema of 2")
                args.bisector = 2

        path_intermed = os.path.join(str(args.intermedfiles), "npz")

        # Create directory for npz output files
        if not os.path.exists(path_intermed):
                os.makedirs(path_intermed)

        filterpars = [int(args.minlinewidth), int(args.maxlinewidth), float(args.minlinedepth), float(args.maxcontdiff),\
                      float(args.mincontavg), float(args.masscenter), float(args.jerkdistance), int(args.bisector)]
                
        # directory for all files
        files = list(pathlib.Path(str(args.filedir)).rglob('*.fits'))
        
        if args.noint:

                # get reference file
                waveref,flux,error,measurements = get_data(files[0], cropL, cropR, args.filetype)
                
                if args.buffer == None:
                        
                        print("normal integration")

                        # initialize 3D array to store all spectra to handle nans properly      
                        big3Darr = np.zeros((len(files), np.shape(waveref)[0], np.shape(waveref)[1]))

                        # error estimate
                        err = np.zeros(len(files))

                        bigarr, err = zip(*tqdm(pool.imap(RefSpec((waveref, cropL, cropR, args.filetype)), files), desc="integrating files"))
                        err = np.asarray(err)
                        print(np.shape(bigarr))
                        print(np.shape(err))
                        
                        # integrate 3D array
                        refspectrum = np.nanmean(bigarr, axis=0)
                        
                        # get error estimate
                        avgerr = np.mean(err[err<1])
                        
                elif (int(args.buffer) > 1) and (int(args.buffer) < len(files)):
                      
                        print("buffered integration")
                      
                        splitlist = np.linspace(0, len(files), 1+int(args.buffer))[1:-1].round().astype(int)
                        filebufs = np.split(files, splitlist)

                        bigarr, err = zip(*tqdm(pool.imap(RefSpecBuf((waveref, cropL, cropR, args.filetype)), filebufs), desc="integrating files"))

                        refspectrum = np.sum(bigarr, axis=0)/len(files)

                        avgerr = np.mean(err)
                else:
                        raise ValueError("buffer input invalid")

                np.savez("refspectrum.npz", waveref, refspectrum, avgerr)

        if args.noproc:

                waveref, csplines, minima, maxima, linedepth, contdiff, contavg, masscenter, jerkdistance, bisectormax, templatemask, temperatures, boxlist =\
                         load_ref_spectrum("refspectrum.npz",'TAPAS_WMKO_NORAYLEIGH_SPEC.fits','TAPAS_WMKO_NORAYLEIGH_SPEC_WVL.fits',
                                           int(args.telluricmaskdepth), float(args.telluricmaskdev), args.templatemask, filterpars[2])

                list(tqdm(pool.imap(FileRV((csplines, minima, maxima, linedepth, contdiff, contavg, masscenter, jerkdistance, bisectormax,\
                        templatemask, boxlist, filterpars, cropL, cropR, args.filetype, path_intermed, args.filecompress)), files), desc="processing files"))
                                      
                # flatten line minima array
                wavelines = np.concatenate(minima)

                # Open npz directory
                files = list(pathlib.Path(path_intermed).glob('*.npz'))

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
                contavg = np.concatenate(contavg)[~dupmask]
                masscenter = np.concatenate(masscenter)[~dupmask]
                jerkdistance = np.concatenate(jerkdistance)[~dupmask]
                bisectormax = np.concatenate(bisectormax)[~dupmask]

                # initialize other arrays
                meansr,means,error,neidrv,time,angle,sindex,mndepth,numlines=np.zeros(len(files)),np.zeros(len(files)),np.zeros(len(files)),np.zeros(len(files)),\
                                                                              np.zeros(len(files)),np.zeros(len(files)),np.zeros(len(files)),np.zeros(len(files)),np.zeros(len(files))

                rvarrays,rverr_arrays,corr_arrays,width_arrays,ind_arrays,depth_arrays,line_search = np.empty(shape=(len(files), len(wavelines))),\
                        np.empty(shape=(len(files), len(wavelines))),np.empty(shape=(len(files), len(wavelines))),np.empty(shape=(len(files), len(wavelines))),\
                        np.empty(shape=(len(files), len(wavelines))),np.empty(shape=(len(files),len(wavelines))),np.empty(shape=(len(files),len(wavelines)))

                # Get high trend RVs in IR and neidrv, time, and solar altitude
                for i in tqdm(range(len(files)), desc="pearson correlation"):

                        arrays = np.load(files[i])

                        rv = arrays["arr_0"][~dupmask]
                        rverr = arrays["arr_1"][~dupmask]

                        if (args.filetype == "neid"):

                                rvred = rv[wavelines > 7000]
                                rverr_red = rverr[wavelines > 7000]

                                cutr = np.where((rverr_red < 3*np.nanmean(rverr_red)) &
                                        (np.abs(rvred - np.nanmean(rvred)) < 3*np.nanstd(rvred)))
                                meansr[i] = np.sum(rvred[cutr]/(rverr_red[cutr]**2))/np.sum(1/(rverr_red[cutr]**2))

                        line_search[i][(rverr < 3*np.nanmean(rverr)) & (np.abs(rv - np.nanmean(rv)) < 3*np.nanstd(rv))] = 1
                        
                        rvarrays[i] = rv
                        rverr_arrays[i] = rverr

                        corr_arrays[i] = arrays["arr_2"][~dupmask]
                        width_arrays[i] = arrays["arr_3"][~dupmask]
                        ind_arrays[i] = arrays["arr_4"][~dupmask]
                        depth_arrays[i] = arrays["arr_5"][~dupmask]
                        measures = arrays["arr_6"]
                        sindex[i],mndepth[i],neidrv[i],time[i],angle[i] = measures

                # Calculate corr coeff for each line with respect to high trend lines for filtering
                pearsoncorr, perline = np.zeros(len(rvarrays.T)),np.zeros(len(rvarrays.T))

                for i in range(len(rvarrays.T)):
                        
                        line = rvarrays.T[i]
                        filelist = line_search[:,i]
                        perline[i] = len(filelist[filelist == 1])/len(filelist)
                        
                        if (args.filetype == "neid"):
                                try:
                                        pearsoncorr[i] = pearsonr(meansr[~np.isnan(line)], line[~np.isnan(line)])[0]
                                except:
                                        pearsoncorr[i] = np.nan

                # Calculate bulk rv
                for i in range(len(files)):
                        
                        rv = rvarrays[i]
                        rverr = rverr_arrays[i]
                        
                        if (args.filetype == "neid"):
                                cut = np.where((rverr < 3*np.nanmean(rverr)) &
                                (np.abs(rv - np.nanmean(rv)) < 3*np.nanstd(rv))& (np.abs(pearsoncorr) < 0.5) & (perline > 0.01))
                        elif ((args.filetype == "harpn") or (args.filetype == "adp")):
                                cut = np.where((rverr < 3*np.nanmean(rverr)) &
                                (np.abs(rv - np.nanmean(rv)) < 3*np.nanstd(rv))& (perline > 0.01))                        

                        numlines[i] = len(rv[cut])

                        means[i] = np.sum(rv[cut]/(rverr[cut]**2))/np.sum(1/(rverr[cut]**2))
                        error[i] = np.mean(rverr[cut])/np.sqrt(len(rv[cut]))

                # Output rvs and calculated parameters
                np.savez("all_lines", rvarrays, rverr_arrays, corr_arrays, width_arrays, ind_arrays, depth_arrays)
                np.savez("filter_pars", wavelines, temperatures, contdiff, contavg, masscenter, jerkdistance, bisectormax, linedepth,pearsoncorr, perline)
                np.savez("output_file", means, error, neidrv, time, angle, sindex, mndepth,numlines, args)
