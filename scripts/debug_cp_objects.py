
import CoolProp.CoolProp as CP
import json
import sys
import os

sys.path.append(os.getcwd())

def check():
    backend = "HEOS"
    fluid = "Methane"
    
    print("--- Initial Get String ---")
    try:
        s = CP.get_fluid_param_string(fluid, "JSON")
        d = json.loads(s)[0]
        print("Initial Transport:", 'transport' in d)
    except Exception as e:
        print("Initial failed:", e)
        
    print("\n--- Creating AbstractState ---")
    asi = CP.AbstractState(backend, fluid)
    asi.update(CP.PT_INPUTS, 101325, 300)
    print("Conductivity:", asi.conductivity())
    
    print("\n--- Get String After AbstractState ---")
    try:
        s = CP.get_fluid_param_string(fluid, "JSON")
        d = json.loads(s)[0]
        print("Transport After:", 'transport' in d)
    except Exception as e:
        print("After failed:", e)

    print("\n--- Instance Method Check ---")
    try:
        s = asi.fluid_param_string("JSON")
        d = json.loads(s)[0]
        print("Transport (Instance Method):", 'transport' in d)
    except Exception as e:
        print("Instance method failed:", e)

if __name__ == "__main__":
    check()
