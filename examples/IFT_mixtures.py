#!/usr/bin/env python
# Created by Darshan on 2025-08-15

from parachorpy import parachor as ST
import numpy as np

if __name__ == '__main__':
    json_files = ["Mulero_2012.json"]
    model = ST.SurfaceTension(json_files)
    model.ctREFPROP_init('~/Software/REFPROP_BETA/REFPROP-cmake/build', gerg_enable=1)

    MIXTURE   = "CO2;Methane"
    z         = [0.95, 0.05]
    T         = 250.0
    PRESSURES = np.arange(20.0, 31.0, 5.0)
    kij       = 0.0

    Pbar, sigma = model.REFPROP_MIXTURE(MIXTURE, z, T, PRESSURES, kij)

    for P, s in zip(Pbar, sigma):
        print(f"P = {P:.1f} bar | sigma = {s:.4f} mN/m")