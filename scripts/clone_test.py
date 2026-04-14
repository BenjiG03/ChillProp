
import CoolProp.CoolProp as CP
import json
import sys
import os

def extract_fluid_params(fluid_name):
    try:
        try:
            fluid_json_str = CP.get_fluid_param_string(fluid_name, "JSON")
            temp_data = json.loads(fluid_json_str)
            if isinstance(temp_data, list): temp_data = temp_data[0]
            
            # Robust check for conductivity data
            trans = temp_data.get('transport', {})
            cond = trans.get('conductivity', {})
            if not cond or 'dilute' not in cond:
                print(f"DEBUG: Transport missing/incomplete for {fluid_name}. Triggering fallback.")
                raise ValueError("Incomplete Transport Data")
            else:
                print("DEBUG: Path 1 - Immediate Success")
                
        except ValueError:
            # Fallback
            backend = 'HEOS'
            try:
                print("DEBUG: Entering Fallback")
                asi = CP.AbstractState(backend, fluid_name)
                asi.update(CP.PT_INPUTS, 101325, 300) 
                
                try:
                    c_val = asi.conductivity()
                    print(f"DEBUG: Forced conductivity calc: {c_val}")
                except Exception as e:
                    print(f"DEBUG: Cond calc failed: {e}")
                
                fluid_json_str = CP.get_fluid_param_string(fluid_name, "JSON")
                
                check = json.loads(fluid_json_str)
                if isinstance(check, list): check = check[0]
                if 'transport' not in check:
                     print(f"DEBUG: Still missing transport. Trying HEOS::{fluid_name}")
                     asi = CP.AbstractState(backend, f"HEOS::{fluid_name}")
                     asi.update(CP.PT_INPUTS, 101325, 300)
                     fluid_json_str = CP.get_fluid_param_string(f"HEOS::{fluid_name}", "JSON")
                else:
                     print("DEBUG: Path 2 - Fallback Success")
            except Exception as e:
                sys.stderr.write(f"Fallback AbstractState failed: {e}\n")
            
        data = json.loads(fluid_json_str)
        if isinstance(data, list): data = data[0]
        
        return data
        
    except Exception as e:
        sys.stderr.write(f"Error extracting parameters for {fluid_name}: {e}\n")
        return None

if __name__ == "__main__":
    fluid = "Methane"
    data = extract_fluid_params(fluid)
    
    if data:
        print(json.dumps(data, indent=2))
    else:
        sys.exit(1)
