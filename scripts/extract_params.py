import CoolProp.CoolProp as CP
import json
import sys
import os

def extract_fluid_params(fluid_name):
    try:
        # CoolProp allows retrieving the JSON structure of the fluid
        # The key is to use get_fluid_param_string
        # Note: sometimes the key is "JSON", sometimes it returns the whole thing.
        # Let's check if it works.
        try:
            fluid_json_str = CP.get_fluid_param_string(fluid_name, "JSON")
        except ValueError:
            # Fallback or maybe the fluid isn't loaded?
            # Creating an AbstractState forces loading.
            backend = 'HEOS'
            asi = CP.AbstractState(backend, fluid_name)
            fluid_json_str = CP.get_fluid_param_string(fluid_name, "JSON")

        data = json.loads(fluid_json_str)
        return data
        
    except Exception as e:
        sys.stderr.write(f"Error extracting parameters for {fluid_name}: {e}\n")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_params.py <FluidName> [OutputFile]")
        sys.exit(1)
        
    fluid = sys.argv[1]
    data = extract_fluid_params(fluid)
    
    if data:
        if len(sys.argv) >= 3:
            with open(sys.argv[2], 'w') as f:
                json.dump(data[0] if isinstance(data, list) else data, f, indent=2)
            print(f"Parameters for {fluid} written to {sys.argv[2]}")
        else:
            print(json.dumps(data[0] if isinstance(data, list) else data, indent=2))
    else:
        sys.exit(1)
