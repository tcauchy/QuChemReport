import matplotlib.pyplot as plt
from quchemreport.utility_services import units


def absoUV(et_energies, et_oscs, xvalues, spectrum, file_name=None):
    # Plotting UV spectrum from the excited states energies and oscillator strengths, the xvalues in wavenumbers and the correspondng absorption in epsilon
    plt.cla()
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # First plot specific lines
    x = xvalues
    y = spectrum[0]
    ax[0].plot(x,y)                                
    ax[0].set(ylabel='$\epsilon$ / L mol$^{-1}$ cm$^{-1}$')
    secax = ax[0].secondary_xaxis('top', functions=(units.nm2wnb, units.wnb2nm)) # Needs Matplotlib 3.1 ! experimental
    secax.set_xlabel('$\lambda$ / nm')
    # Second plot specific lines
    x2 = et_energies
    y2 = et_oscs
    ax[1].axhline(y=0,color='black')
    ax[1].bar(x2,y2,width=250)
    ax[1].set(ylabel='Oscillator strength')
    # Common lines for the 2 plots
    plt.gca().invert_xaxis()                        
    plt.xlabel('Wavenumber / cm$^{-1}$')
    plt.minorticks_on()
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.savefig("{}-UV-Abso-Spectrum.png".format(file_name), dpi=300)
    #plt.show()                

def emiUV(et_energies, et_oscs, xvalues, spectrum, file_name=None):
    # Plotting UV spectrum from the excited states energies and oscillator strengths, the xvalues in wavenumbers and the correspondng absorption in epsilon
    plt.cla()
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # First plot specific lines
    x = xvalues
    y = spectrum[0]
    ax[0].plot(x,y)                                
    ax[0].set(ylabel='molar emission / L mol$^{-1}$ cm$^{-1}$')
    secax = ax[0].secondary_xaxis('top', functions=(units.nm2wnb, units.wnb2nm)) # Needs Matplotlib 3.1 ! experimental
    secax.set_xlabel('$\lambda$ / nm')
    # Second plot specific lines
    x2 = et_energies
    y2 = et_oscs
    ax[1].axhline(y=0,color='black')
    ax[1].bar(x2,y2,width=250)
    ax[1].set(ylabel='Oscillator strength')
    # Common lines for the 2 plots
    plt.gca().invert_xaxis()                        
    plt.xlabel('Wavenumber / cm$^{-1}$')
    plt.minorticks_on()
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.savefig("{}-UV-Emi-Spectrum.png".format(file_name), dpi=300)
    #plt.show()                


def absoCD(et_energies, et_rotats, xvalues, CDspectrum, file_name=None):
    # Plotting CD spectrum from the excited states energies and rotational strengths, the xvalues in wavenumbers and the correspondng absorption in epsilon
    plt.cla()
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # First plot specific lines
    x = xvalues
    y = CDspectrum[0]
    ax[0].plot(x,y)                                
    ax[0].axhline(y=0,color='black',ls='--')
    ax[0].set(ylabel='$\Delta \epsilon$ / L mol$^{-1}$ cm$^{-1}$')
    secax = ax[0].secondary_xaxis('top', functions=(units.nm2wnb, units.wnb2nm)) # Needs Matplotlib 3.1 ! experimental
    secax.set_xlabel('$\lambda$ / nm')
    # Second plot specific lines
    x2 = et_energies
    y2 = et_rotats
    ax[1].axhline(y=0,color='black')
    ax[1].bar(x2,y2,width=250)
    ax[1].set(ylabel='Rotational strength')
    # Common lines for the 2 plots
    plt.gca().invert_xaxis()                        
    plt.xlabel('Wavenumber / cm$^{-1}$')
    plt.minorticks_on()
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.savefig("{}-UV-CD-Spectrum.png".format(file_name), dpi=300)
    #plt.show()                

def emiCD(et_energies, et_rotats, xvalues, CDspectrum, file_name=None):
    # Plotting CD spectrum from the excited states energies and rotational strengths, the xvalues in wavenumbers and the correspondng absorption in epsilon
    plt.cla()
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # First plot specific lines
    x = xvalues
    y = CDspectrum[0]
    ax[0].plot(x,y)                                
    ax[0].axhline(y=0,color='black',ls='--')
    ax[0].set(ylabel='$\Delta molar emission / L mol$^{-1}$ cm$^{-1}$')
    secax = ax[0].secondary_xaxis('top', functions=(units.nm2wnb, units.wnb2nm)) # Needs Matplotlib 3.1 ! experimental
    secax.set_xlabel('$\lambda$ / nm')
    # Second plot specific lines
    x2 = et_energies
    y2 = et_rotats
    ax[1].axhline(y=0,color='black')
    ax[1].bar(x2,y2,width=250)
    ax[1].set(ylabel='Rotational strength')
    # Common lines for the 2 plots
    plt.gca().invert_xaxis()                        
    plt.xlabel('Wavenumber / cm$^{-1}$')
    plt.minorticks_on()
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.savefig("{}-UV-CD-Emi-Spectrum.png".format(file_name), dpi=300)
    #plt.show()                





            # Calculating of IR spectrum
            #GaussSum_IRspectrum.Vibfreq(jf[i],os.path.basename(lf[i]),600,4000,1700,6.0,"Gen",1,0,jf[i]["comp_details"]["freq"]["temperature"])
        # Plotting IR spectrum 
        #extracts the datas from the resulting txt file to generate a png file of the spectrum
#            x,y = np.loadtxt("IRSpectrum.txt",skiprows=2,usecols=(0,1), unpack=True)
#            plt.xlabel('Wavenumber (cm-1)')
#            plt.ylabel('IR activity')
#            plt.title("Calculated Infrared Spectrum")
#            plt.gca().invert_yaxis()
#            plt.gca().invert_xaxis()
#            #plt.plot(x,y, 'o', color='black',linestyle="-",markersize= 2 )
#            plt.plot(x,y, color='blue',linestyle="-" )
#            fig = plt.gcf()
#            fig.set_size_inches(11.5, 6.5)
#            plt.savefig("img-IR_Spectrum.png", dpi=300)
            # plt.show()
