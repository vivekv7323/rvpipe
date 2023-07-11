import os
import numpy as np
from astropy.io import fits
from numpy.core.multiarray import interp
#from PyAstronomy import pyasl
import matplotlib.pyplot as plt
#from timeit import default_timer as timer
from scipy import stats
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_filter1d
from tqdm import tqdm

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
        data_spec = hdul[1].data[9:-5,:]
        # the actual blaze (in flux units)
        blz_spec = hdul[15].data[9:-5,:]
        # the variance of the spectrum
        # var = fits.getdata(path, fits_extension_variance)[9:-5,:]
        var = np.ones_like(data_spec)
        var[data_spec>0] = np.sqrt(data_spec[data_spec>0])
        # the wavelength solution of the spectrum (natively in Angstroms)
        wsol = hdul[7].data[9:-5,:] # A
        # Shift to heliocentric frame, compensate for zero point offset
        wsol = wsol + wsol * (berv-83285) / 299792458
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
            X.append( x ); Y.append( y/b ); E.append( np.sqrt(e)/b )

        ''' 
        wavelength = X[0];flux = Y[0];error = E[0]
        for i in range(len(X)-1):
            # indices without overlap
            num1 = np.where(wavelength < (wavelength[-1]+X[i+1][0])/2)
            num2 = np.where(X[i+1] > (wavelength[-1]+X[i+1][0])/2)

            # concatenate with main array
            wavelength = np.concatenate((wavelength[num1],X[i+1][num2]))
            flux = np.concatenate((flux[num1], Y[i+1][num2]))
            error = np.concatenate((error[num1], E[i+1][num2]))
            
        num3 = np.where((wavelength>3900) & (wavelength<5850))
        return wavelength[num3], flux[num3], error[num3], berv
        '''

        # convert to numpy arrays
        X = np.array(X)
        Y = np.array(Y)
        E = np.array(E)
        return X, Y, E, berv

'''
creates reference spectrum from files in path, and a wavelength reference
'''
def createRefSpectrum(path, reference):

        # directory for all files
        files = os.listdir(path)

        # get reference file
        waveRef,flux,error,berv = fetch_single(reference)

        # get NEID measurements for reference
        neidrvRef = fits.open(reference)[12].header['CCFRVMOD']*1000
        #waveRef += waveRef * neidrvRef/299792458

        # initialize array for reference spectrum
        refSpectrum = np.zeros(np.shape(waveRef))

        #if not os.path.exists('normFlux'):
        #      os.makedirs('normFlux')      

        # integrate fluxes for all files after interpolating them to the same wavelength array
        for file in tqdm(files, desc="integrating reference"):
                
            wavelength, flux, error, berv = fetch_single(path + '/' + file)
            
            for i in range(len(flux)):
                flux[i] = flux[i]/maximum_filter1d(np.where(np.isnan(flux[i]),-np.inf, flux[i]), size=1000)
            #np.savez('normFlux/'+files[i][:-5]+'_norm', flux)
            
            neidrv = fits.open(path + '/' + file)[12].header['CCFRVMOD']*1000
            #wavelength += wavelength * neidrv/299792458
            
            refSpectrum += np.concatenate([interp(waveRef[[i]],wavelength[i],flux[i]) for i in range(np.shape(flux)[0])])

        return waveRef, refSpectrum/len(files)

'''
load reference spectrum, create telluric mask and line windows
'''
def loadRefSpectrum(path, startA, endA):
       
        # Load reference spectrum
        result = np.load(path)
        wavelength = result["arr_0"]
        flux = result["arr_1"]

        bigMask = (wavelength<0)
        for i in tqdm(range(len(startA)), desc="constructing telluric mask"):
            bigMask |= ((wavelength>(startA[i]-0.4))&(wavelength<(endA[i]+0.4)))

        #cSplines = list([]) for i in range(np.shape(flux)[0])])

        '''
        Get local minima for absorption lines, get cubic spline model,       
        get local maxima to set windows for each line 
        '''
        preTelMinima = []
        minima = []
        maxima = []
        cSplines = []
        for i in range(np.shape(flux)[0]):
            #flux[i] = flux[i]/maximum_filter1d(np.where(np.isnan(flux[i]),-np.inf, flux[i]), size=1000)
            cSplines.append(CubicSpline(wavelength[i][np.isfinite(flux[i])], flux[i][np.isfinite(flux[i])]))
            maxima.append(find_peaks(flux[i], distance=1,height=.02, prominence=.02)[0])
            minList = find_peaks(np.nanmax(flux[i])-flux[i], distance=5,height=.1, prominence=.1)[0]
            preTelMinima.append(minList)
            minima.append(minList[np.isin(minList, np.where(bigMask[i] == False)[0])])

        contDiff = []
        lineDepth = []
##        massCenter = []
##        smallWindow = []
##        jerkDistance = []
##        bisector = []
##        numMaxima = []
        boxList = []

        # create line windows
        for i in tqdm(range(len(minima)),desc="creating line windows"):
                
                #, massCenterOrd, swOrd, jDOrd, bisectorOrd, numMaximaOrd,
                contDiffOrd, lineDepthOrd, boxOrd = np.zeros(len(minima[i])), np.zeros(len(minima[i])),np.zeros(len(minima[i]))#,np.zeros(len(minima[i]))
                   #np.zeros(len(minima[i])),np.zeros(len(minima[i])),np.zeros(len(minima[i])),np.zeros(len(minima[i])),np.zeros(len(minima[i]))
                
                for j in range(len(minima[i])):

                        # get local maxima surrounding line
                        nearestMaxIndex = np.argmin(np.abs(maxima[i] - minima[i][j]))
                        if (wavelength[i][maxima[i][nearestMaxIndex]] >  wavelength[i][minima[i][j]]) & (nearestMaxIndex != 0):
                            otherMaxOrd = flux[i][maxima[i][nearestMaxIndex - 1]]
                        elif (wavelength[i][maxima[i][nearestMaxIndex]] <  wavelength[i][minima[i][j]]) & (nearestMaxIndex != (len(maxima[i])-1)):
                            otherMaxOrd = flux[i][maxima[i][nearestMaxIndex + 1]]
                        else:
                            otherMaxOrd = np.nan

                        # difference between maxima
                        contDiffOrd[j] = np.abs(otherMaxOrd - flux[i][maxima[i][nearestMaxIndex]])

                        # box around window
                        boxOrd[j] = np.abs(wavelength[i][maxima[i][nearestMaxIndex]] - wavelength[i][minima[i][j]])

                        # Get location of line peak
                        lineMin = wavelength[i][minima[i][j]]

                        lineDepthOrd[j] = 1 - 2*flux[i][minima[i][j]]/(otherMaxOrd+flux[i][maxima[i][nearestMaxIndex]])

##                        # indices of the window 
##                        indicesOrd.append(np.where((wavelength[i] < (box+lineMin)) & (wavelength[i] > (lineMin-box))))

                contDiff.append(contDiffOrd)
                #indicesList.append(indicesOrd)
                boxList.append(boxOrd)
                lineDepth.append(lineDepthOrd)

        return cSplines, minima, maxima, contDiff, lineDepth, boxList, preTelMinima

'''
create arrays with telluric line groups ot mask out
'''
def createTelluricArrays(telPath, wvlPath):

        y = fits.open(telPath)[0].data        
        x = fits.open(wvlPath)[0].data * 10 # angstrom

        # smooth out a bit to get rid of continuum
        y = y/maximum_filter1d(y, size=2)
        mask = np.where((np.abs(y-1) > 1e-4))[0]

        # create groups
        start = x[mask[np.where(np.diff(mask) != 1)]]
        end = x[mask[np.where(np.diff(mask) != 1)[0]+1]]
        start = np.insert(start, len(start), x[mask[-1]])
        end = np.insert(end, 0, x[mask[0]])    

        return start,end

'''
process single file
'''
def singleFileRV(path, cSplines, minima, maxima, boxList):

        # Currently working on single file, looping all files later
        wS, fS, eS, bervS = fetch_single(path)

        # Interpolate flux of reference to file
        flux = np.vstack([cSplines[i](wS[i]) for i in range(np.shape(wS)[0])])

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
                Sindex = (CaHflux2+CaKflux2)/(flux3900+flux4000)
                #RHKprime = pyasl.SMW_RHK(ccfs='noyes', afc='middelkoop', rphot='noyes').SMWtoRHK(Sindex, 5778, 0.656, lc='ms', verbose=False)[0]

                # Get Mn I wavelengths
                Mn1 = np.where((wS[50][minima[50]] > 5394) & (wS[50][minima[50]] < 5396))[0]
                Mn2 = np.where((wS[51][minima[51]] > 5394) & (wS[51][minima[51]] < 5396))[0]
        except:
                raise ValueError('masking obscures critical lines')
        
        #print(CaH, CaK)
        #print(wS[50][minima[50][Mn1]][0],wS[51][minima[51][Mn2]][0])

        # Calculate RV, RV errors, and other parameters
        RV = []
        RVError = []
        corrCoeff = []
        lineWidth = []
        Mnlinedepth = 0

        # calculate RVs for each order
        for i in range(len(minima)):
                
            # continuum division
            fluxCont = fS[i]/maximum_filter1d(np.where(np.isnan(fS[i]),-np.inf, fS[i]), size=1000)
            errorCont = eS[i]/maximum_filter1d(np.where(np.isnan(fS[i]),-np.inf, fS[i]), size=1000)
            # Initialize arrays
            RVOrd, RVErrorOrd, corrCoeffOrd, lineWidthOrd = np.zeros(len(minima[i])),np.zeros(len(minima[i])),np.zeros(len(minima[i]))\
                                                                          ,np.zeros(len(minima[i]))

            if ((i==50) | (i==51)):
                mnbox = np.abs(wS[i][maxima[i]][np.argmin(np.abs(wS[i][maxima[i]] - 5394.7))] - 5394.7)
                relFlux = fluxCont[(wS[i] > (5394.7-mnbox)) & (wS[i] < (5394.7+mnbox))]
                Mnlinedepth += 1 - (np.min(relFlux))/np.max(relFlux)

            # iterate over lines in each order
            for j in range(len(minima[i])):
                        
                box = boxList[i][j]
                lineMin = wS[i][minima[i][j]]
                indices = np.where((wS[i] < (box+lineMin)) & (wS[i] > (lineMin-box)))
                fluxSpec = fluxCont[indices]
                # minimum pixel length of window
                if ((len(indices[0]) > 10) & (len(indices[0]) < 100) & (~np.isnan(fluxSpec).any())):
                        # get wavelength of interpolated target spectrum in the window
                        waveLine = wS[i][indices]
                        der = cSplines[i](waveLine, 1)
                        #lineDepthOrd[j] = 1 - 
                        halfMax = 0.5*(np.min(fluxSpec)+np.max(fluxSpec))
                        lineWidthOrd[j] = np.abs(waveLine[waveLine < lineMin][np.argmin(np.abs(fluxSpec[waveLine < lineMin] - halfMax))] -\
                                          waveLine[waveLine > lineMin][np.argmin(np.abs(fluxSpec[waveLine > lineMin] - halfMax))])
                        # try/except linear regression
                        location = ~np.isnan(flux[i][indices]) & ~np.isnan(fluxSpec)
                        # linear least-squares regression
                        try:
                                def model(S,A,Adl):
                                        return A * S + Adl*der[location]
                                bestpars,parsCov = curve_fit(model,flux[i][indices][location],\
                                                             fluxSpec[location], sigma = errorCont[indices][location], absolute_sigma=True)
##                                covInv = np.linalg.inv(np.diag(errorCont[indices][location]**2))
##                                designMatrix = np.vstack((flux[i][indices][location], cSplines[i](waveLine[location], 1)))
##                                parsCov = np.linalg.inv(designMatrix @ covInv @ designMatrix.T)
##                                bestpars = parsCov @ designMatrix @ covInv @ fluxSpec[location]
                                corrCoeffOrd[j] = np.sqrt(np.square(parsCov[1][0])/(parsCov[1][1]*parsCov[0][0]))
                                RVOrd[j] = (299792458*bestpars[1]/(bestpars[0]*lineMin))
                                RVErrorOrd[j] = (299792458/lineMin)*np.sqrt((parsCov[1][1]/bestpars[0]**2)\
                                                                         + (np.sqrt(parsCov[0][0])*bestpars[1]/(bestpars[0]**2))**2)
                        except:
                                #print(np.min(np.abs(maximum_filter1d(fS[i], size=2000))))
                                raise ValueError('linear regression failed')
                                
                else:
                        RVOrd[j] = np.nan
                        RVErrorOrd[j] = np.nan
                        corrCoeffOrd[j] = np.nan
                        lineWidthOrd[j] = np.nan
                        
            RV.append(RVOrd)
            RVError.append(RVErrorOrd)
            corrCoeff.append(corrCoeffOrd)
            lineWidth.append(lineWidthOrd)
            
        return RV, RVError, corrCoeff, lineWidth, Sindex, Mnlinedepth*0.5

waveRef, refSpectrum = createRefSpectrum('data', 'data/neidL2_20220323T163236.fits')
# Save reference spectrum
np.savez("refSpectrum.npz", waveRef, refSpectrum)

# directory for all files
files = os.listdir('data')
startA, endA = createTelluricArrays('TAPAS_WMKO_NORAYLEIGH_SPEC.fits', 'TAPAS_WMKO_NORAYLEIGH_SPEC_WVL.fits')
cSplines, minima, maxima, contDiff, lineDepth, boxList, preTelMinima = loadRefSpectrum("refSpectrum.npz",startA,endA)

if not os.path.exists('npz'):
      os.makedirs('npz')   

Sindices,Mndepths = np.zeros(len(files)),np.zeros(len(files))

#for i in range(len(files)):
for i in tqdm(range(len(files)), desc="Processing files"):
        #print(files[i][:-5])
        RV, RVError, corrCoeff, lineWidth, Sindex, Mnlinedepth = singleFileRV('data' + '/' + files[i], cSplines, minima, maxima, boxList)
        np.savez("npz"+'/'+ files[i][:-5]+"_RV", RV, RVError, corrCoeff, lineWidth)  
        Sindices[i] = Sindex
        Mndepths[i] = Mnlinedepth

np.savez("otherParams", minima, contDiff, lineDepth, Sindices, Mndepths, preTelMinima, boxList)
