
import CoolProp
import os
import json

def find_file():
    cp_path = os.path.dirname(CoolProp.__file__)
    print(f"CoolProp Path: {cp_path}")
    
    # Usual structure: CoolProp/type/Fluids/HEOS/Methane.json
    # Or just share/CoolProp/fluids/HEOS...
    
    # Recursive search
    matches = []
    for root, dirs, files in os.walk(cp_path):
        if "Methane.json" in files:
            matches.append(os.path.join(root, "Methane.json"))
            
    print(f"Matches: {matches}")
    
    if matches:
        target = matches[0]
        print(f"Reading {target}...")
        with open(target, 'r') as f:
            data = json.load(f)
            # Unwrap if list
            if isinstance(data, list): data = data[0]
            print("Transport in File:", 'transport' in data)

if __name__ == "__main__":
    find_file()
