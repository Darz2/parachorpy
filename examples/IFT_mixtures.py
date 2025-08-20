#!/usr/bin/env python
# Created by Darshan on 2025-08-15

from parachorpy import parachor as IFT
import numpy as np

if __name__ == '__main__':
    
    json_files  = ["Mulero_2012.json"]
    model       = IFT.InterfacialTension(json_files)
    model.ctREFPROP_init('~/Software/REFPROP_BETA/REFPROP-cmake/build', gerg_enable=1)

    MIXTURE   = "CO2;Methane"
    z         = [0.95, 0.05]
    T         = 250.0
    PRESSURES = np.arange(20.0, 31.0, 5.0)
    kij       = 0.0

    Pbar, gamma = model.REFPROP_MIXTURE(MIXTURE, z, T, PRESSURES, kij)

    for P, s in zip(Pbar, gamma):
        print(f"P = {P:.1f} bar | gamma = {s:.4f} mN/m")