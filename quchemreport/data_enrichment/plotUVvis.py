import math 
import numpy as np

def GaussianSpectrum(start,end,numpts,et_energies,heights,FWHM):
## Calculate the absorption spectra with a gaussian broadening. Adapted from GaussSum program
# et_energies is a list of excited states energies in cm-1
# et_oscs is a list of oscillator strength

    peaks = et_energies
    xvalues = np.arange(numpts)*float(end-start)/(numpts-1) + start
    data = []
    A = -2.7726/FWHM**2
    for x in xvalues:
        tot = [0]*len(heights) # Sum of each gaussian bands for this x value

        for peakno in range(len(peaks)): # For each peak
            pos = peaks[peakno]
            exponent = math.exp(A*(pos-x)**2)
            for spectrumno in range(len(heights)):
                tot[spectrumno] += heights[spectrumno][peakno]*exponent

        data.append(tot)

    spectrum = np.swapaxes(np.array(data),0,1)
    return xvalues, spectrum
