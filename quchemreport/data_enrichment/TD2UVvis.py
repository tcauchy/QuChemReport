# Transform the Time dependent calculations into spectra with a gaussian broadening.

# This is a modified version of the GaussSum code to generate the UV Spectra
# N. M. O'Boyle, A. L. Tenderholt and K. M. Langner. J. Comp. Chem., 2008, 29, 839-845.

# This program is free software; you can redistribute and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY, without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

import math 
import numpy as np

def GaussianSpectrum(start,end,numpts,et_energies,heights,FWHM):
# Calculate the absorption spectra with a gaussian broadening. Adapted from the GaussSum program
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

def CDheights(et_energies, et_rotats, FWHM):
    # Equation 8 in Stephens, Harada, Chirality, 2010, 22, 229.
    # This uses Delta, the half width at 1/e height.
    # We use Sigma, the full width at 1/e height.
    #   Delta = Sigma / 2
    sigma = FWHM / math.sqrt(math.log(2))
    Delta = sigma / 2.
    prefactor = 1.0 / (2.296e-39 * math.sqrt(math.pi) * Delta)
    heights = [] 
    for x_cd in range(len(et_rotats)):
        heights.append(prefactor * et_rotats[x_cd] * et_energies[x_cd] * 1e-40)
    return heights
