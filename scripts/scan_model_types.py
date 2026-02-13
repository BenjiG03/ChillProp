import json
import os
from collections import defaultdict, Counter

def scan_fluids():
    fluids_dir = r"c:\Users\Benji\Documents\ChillProp\CoolProp\dev\fluids"
    json_files = [f for f in os.listdir(fluids_dir) if f.endswith(".json")]
    
    model_types = defaultdict(Counter)
    
    print(f"Scanning {len(json_files)} fluid files...")
    
    for filename in json_files:
        path = os.path.join(fluids_dir, filename)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            # 1. Ancillaries
            if "ANCILLARIES" in data:
                for key, val in data["ANCILLARIES"].items():
                    if isinstance(val, dict) and "type" in val:
                        model_types[f"ANCILLARIES.{key}"].update([val["type"]])
            
            # 2. EOS Alpha0
            if "EOS" in data and isinstance(data["EOS"], list) and len(data["EOS"]) > 0:
                eos = data["EOS"][0]
                if "alpha0" in eos:
                    for term in eos["alpha0"]:
                        if "type" in term:
                            model_types["EOS.alpha0"].update([term["type"]])
                            
                if "alphar" in eos:
                    for term in eos["alphar"]:
                        if "type" in term:
                            model_types["EOS.alphar"].update([term["type"]])
                            
            # 3. Transport
            if "TRANSPORT" in data:
                trans = data["TRANSPORT"]
                # Viscosity
                if "viscosity" in trans:
                    visc = trans["viscosity"]
                    for key in ["dilute", "initial_density", "higher_order", "residual"]: # Check diverse sub-models
                        if key in visc and "type" in visc[key]:
                            model_types[f"TRANSPORT.viscosity.{key}"].update([visc[key]["type"]])
                        elif key in visc and "type" not in visc[key]:
                             # Sometimes it's nested or implicit?
                             pass

                # Conductivity
                if "conductivity" in trans:
                    cond = trans["conductivity"]
                    for key in ["dilute", "critical", "residual"]:
                         if key in cond and "type" in cond[key]:
                            model_types[f"TRANSPORT.conductivity.{key}"].update([cond[key]["type"]])

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Report
    print("\n=== Model Type Report ===\n")
    for category, counts in sorted(model_types.items()):
        print(f"--- {category} ---")
        for mtype, count in counts.most_common():
            # Find example fluids (inefficient but works for small N)
            examples = []
            for filename in json_files:
                path = os.path.join(fluids_dir, filename)
                try:
                    with open(path, "r") as f:
                        txt = f.read()
                        if mtype in txt and category.split('.')[-1] in txt: # Heuristic
                             examples.append(filename)
                except: pass
            
            print(f"  {mtype}: {count} (e.g., {examples[:3]})")
        print()

if __name__ == "__main__":
    scan_fluids()
