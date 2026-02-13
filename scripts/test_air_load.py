import sys
import os
sys.path.append(os.getcwd())

from chillprop.highlevel import get_params
try:
    params = get_params("Air")
    print(f"Successfully loaded params for {params.name}")
    print(f"Tc: {params.Tc}")
    print(f"Keywords in viscosity: {params.viscosity.dilute if params.viscosity else 'None'}")
    print(f"Keywords in conductivity: {params.conductivity.dilute if params.conductivity else 'None'}")
except Exception as e:
    print(f"Failed to load Air: {e}")
    import traceback
    traceback.print_exc()
