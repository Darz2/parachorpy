#!/usr/bin/env python
# Created by Darshan on 2026-01-09

#!/usr/bin/env python
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import os

lib_path  = os.path.expanduser("~/Software/REFPROP/REFPROP-cmake/build")
data_path = os.path.expanduser("~/Software/REFPROP/REFPROP-cmake/build")

RP = REFPROPFunctionLibrary(lib_path)
RP.SETPATHdll(data_path)

print("Version:", RP.RPVersion())

res = RP.REFPROPdll("CO2", "TP", "D", 0, 0, 0, 300.0, 10, [1.0])
print("ierr:", res.ierr)
print("herr:", res.herr)
print("D:", res.Output[0])
