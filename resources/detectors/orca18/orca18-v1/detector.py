"""Static ORCA-18 detector loader for SIREN (compat shim).

The new SIREN resource API expects each detector model folder to expose a
detector.py with a load_detector() function returning a built DetectorModel.
This file vendors that loader for ORCA-18, reusing the materials/densities
.dat files that ship with the KM3NeTORCA model.
"""
import os
import siren

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_detector(*args, **kwargs):
    det = siren.detector.DetectorModel()
    det.LoadMaterialModel(os.path.join(_HERE, "materials.dat"))
    det.LoadDetectorModel(os.path.join(_HERE, "densities.dat"))
    return det
