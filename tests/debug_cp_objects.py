
import CoolProp.CoolProp as CP
import json

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
    
    print("\n--- AbstractState Methods ---")
    methods = dir(asi)
    print([m for m in methods if 'param' in m.lower() or 'json' in m.lower() or 'string' in m.lower()])
    
    print("\n--- Instance Method Check ---")
    try:
        s = asi.fluid_param_string("JSON")
        d = json.loads(s)[0]
        print("Transport (Instance Method):", 'transport' in d)
    except Exception as e:
        print("Instance method failed:", e)

    print("\n--- Get String After AbstractState ---")
    try:
        s = CP.get_fluid_param_string(fluid, "JSON")
        d = json.loads(s)[0]
        print("Transport After:", 'transport' in d)
    except Exception as e:
        print("After failed:", e)

if __name__ == "__main__":
    check()
