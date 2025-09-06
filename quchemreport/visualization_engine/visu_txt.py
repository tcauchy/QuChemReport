from quchemreport.utility_services import units

def textfile_UV(xvalues, spectrum, file_name=None):
    # Creating a text file of the UV-vis absorption spectrum 
    outputfile=open("{}UV-Abso-Spectrum.txt".format(file_name),"w")
    outputfile.write(" cm-1    nm    Epsilon (L.mol-1.cm-1)    \n")
    x_index = 0 
    for x in xvalues:
        outputfile.write("%d    %.1f         %d \n" %(x, units.wnb2nm(x), spectrum[0,x_index]))
        x_index += 1
    outputfile.write("\n")
    outputfile.close()                                                 


def textfile_CD(xvalues, spectrum, CDspectrum, file_name=None):
    # Creating a text file of the UV-vis absorption and circular dichroism spectra 
    outputfile=open("{}UV-Abso-CD-Spectrum.txt".format(file_name),"w")
    outputfile.write(" cm-1    nm    Epsilon (L.mol-1.cm-1)  Delta-Epsilon (L.mol-1.cm-1)    \n")
    x_index = 0 
    for x in xvalues:
        outputfile.write("%d    %.1f         %d               %.2f\n" %(x, units.wnb2nm(x), spectrum[0,x_index], CDspectrum[0,x_index]))
        x_index += 1
    outputfile.write("\n")
    outputfile.close()  

def textfile_emiUV(xvalues, spectrum, file_name=None):
    # Creating a text file of the UV-vis absorption spectrum 
    outputfile=open("{}UV-Emi-Spectrum.txt".format(file_name),"w")
    outputfile.write(" cm-1    nm    Epsilon (L.mol-1.cm-1)    \n")
    x_index = 0 
    for x in xvalues:
        outputfile.write("%d    %.1f         %d \n" %(x, units.wnb2nm(x), spectrum[0,x_index]))
        x_index += 1
    outputfile.write("\n")
    outputfile.close()                                                 


def textfile_emiCD(xvalues, spectrum, CDspectrum, file_name=None):
    # Creating a text file of the UV-vis absorption and circular dichroism spectra 
    outputfile=open("{}UV-Abso-CD-Emi-Spectrum.txt".format(file_name),"w")
    outputfile.write(" cm-1    nm    Epsilon (L.mol-1.cm-1)  Delta-Epsilon (L.mol-1.cm-1)    \n")
    x_index = 0 
    for x in xvalues:
        outputfile.write("%d    %.1f         %d               %.2f\n" %(x, units.wnb2nm(x), spectrum[0,x_index], CDspectrum[0,x_index]))
        x_index += 1
    outputfile.write("\n")
    outputfile.close()  
