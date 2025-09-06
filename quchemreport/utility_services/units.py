## -*- encoding: utf-8 -*-

## Conversion factors 

import numpy as np


def nm2wnb(x):
    # x in nanometers, returns the equivalent in wavenumber (cm-1)
    return 10000000.0 / x

def wnb2nm(x):
    # TODO : TO CHECK (CHECK IF CORRECT, RAISE AN ERROR IF 0 STOP THE EXEC)
    # Remplace zeros by NaN to avoid division by zero
    x = np.array(x, dtype=float)
    X = np.where(x == 0, np.nan, x)
    
    # x in wavenumber (cm-1), returns the equivalent in nanometers
    return 10000000.0 / X

## Hartree to eV conversion factor
Eh_to_eV = 27.21138505

## Hartree to cm-1 conversion factor
Eh_to_cm_1 = 219474.6313702

## Hartree to Kcal.mol^-1 conversion factor
Eh_to_Kcal_mol = 627.510

## A.U. to Debye conversion factor
ea0_to_D = 0.393430307

## eA (electron*Angstrom) to Debye conversion factor
eA_to_D = 0.2081943

## Angstrom to Bohr conversion factor
A_to_a0 = 0.52917721092

## Picometre-Bohr conversion factor
pm_to_a0 = A_to_a0*100

## Nanometers to wavenumber conversion
nm_to_wnb = 10000000.0


