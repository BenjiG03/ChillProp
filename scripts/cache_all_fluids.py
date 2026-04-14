
import json
import os
import sys
import CoolProp.CoolProp as CP

def get_fluid_json(fluid):
    print(f"Extracting {fluid}...")
    # Force load
    try:
        backend = "HEOS"
        asi = CP.AbstractState(backend, fluid)
        asi.update(CP.PT_INPUTS, 101325, 300)
        # Force transport loading
        try: asi.conductivity()
        except: pass
        try: asi.viscosity()
        except: pass
    except Exception as e:
        print(f"Warning: Could not force load {fluid}: {e}")

    # Now get string
    try:
        s = CP.get_fluid_param_string(fluid, "JSON")
        data = json.loads(s)
        if isinstance(data, list): data = data[0]
        
        # Check transport
        if 'transport' not in data:
            print(f"Warning: {fluid} JSON still missing transport data.")
        else:
            print(f"Success: {fluid} has transport data.")
            
        return data
    except Exception as e:
        print(f"Error: {fluid} extraction failed: {e}")
        return None

fluids = [
    "Nitrogen", "Oxygen", "Argon", "Air", 
    "Hydrogen", "CarbonDioxide", "Water",
    "Methane", "Ethane", "Propane", 
    "n-Butane", "IsoButane", "n-Dodecane"
]

data_dir = "chillprop/data"
os.makedirs(data_dir, exist_ok=True)

for f in fluids:
    data = get_fluid_json(f)
    if data:
        with open(os.path.join(data_dir, f"{f}.json"), 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=2)
        print(f"Saved {f}.json")
