
import sys
import os
sys.path.append(os.getcwd())
import json

# Minimal imports to test isolation
import CoolProp.CoolProp as CP
from scripts.extract_params import extract_fluid_params

def check_methane():
    fluid = "Methane"
    print(f"Checking {fluid} Extraction Isolation...")
    
    try:
        data = extract_fluid_params(fluid)
        cond = data.get('transport', {}).get('conductivity', {})
        print(f"Transport/Conductivity found? {bool(cond)}")
        print(f"Dilute? {'dilute' in cond}")
        print(f"Residual? {'residual' in cond}")
    except Exception as e:
        print(f"Extraction failed: {e}")

if __name__ == "__main__":
    check_methane()
