import sys
import os
import traceback
sys.path.append(os.getcwd())

from chillprop.highlevel import PropsSI

def debug_co2():
    fluid = "CarbonDioxide"
    print(f"Attempting to load {fluid}...")
    try:
        # T, P check
        rho = PropsSI("D", "T", 300, "P", 1e6, fluid)
        print(f"Density: {rho}")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    debug_co2()
