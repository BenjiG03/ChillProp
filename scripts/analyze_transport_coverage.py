
import json
import glob
import os
from collections import Counter

def analyze_transport():
    files = glob.glob(r"c:\Users\Benji\Documents\ChillProp\CoolProp\dev\fluids\*.json")
    
    visc_dilute = Counter()
    visc_initial = Counter()
    visc_ho = Counter()
    
    cond_dilute = Counter()
    cond_residual = Counter()
    cond_critical = Counter()
    
    hardcoded_fluids = []
    
    for fpath in files:
        fname = os.path.basename(fpath).replace('.json', '')
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            # Helper to get first item of list if list
            fluid_data = data[0] if isinstance(data, list) else data
            
            trans = fluid_data.get('TRANSPORT', {})
            
            # Viscosity
            v = trans.get('viscosity', {})
            if not v:
                visc_dilute['None'] += 1
            else:
                # Dilute
                d = v.get('dilute', {})
                if 'hardcoded' in d:
                    visc_dilute['hardcoded'] += 1
                    hardcoded_fluids.append(f"{fname} (Visc Dilute)")
                elif isinstance(d, dict):
                    t = d.get('type', 'Unknown')
                    visc_dilute[t] += 1
                    if t == 'Unknown':
                        print(f"{fname} Visc Dilute Unknown: {d.keys()}")
                else: 
                    visc_dilute['Unknown'] += 1
                    
                # Initial
                i = v.get('initial_density', {})
                if i:
                    visc_initial[i.get('type', 'Unknown')] += 1
                else:
                    visc_initial['None'] += 1
                    
                # Higher Order
                h = v.get('higher_order', {})
                if h:
                    if 'hardcoded' in h:
                        visc_ho['hardcoded'] += 1
                        print(f"{fname}: Hardcoded Higher Order")
                    elif isinstance(h, dict):
                        t = h.get('type', 'Unknown')
                        visc_ho[t] += 1
                        if t == 'friction_theory':
                             print(f"{fname}: Friction Theory")
                        elif t == 'Unknown':
                             print(f"{fname} Visc HO Unknown: {h.keys()}")
                else:
                    visc_ho['None'] += 1
            
            # Conductivity
            c = trans.get('conductivity', {})
            if not c:
                cond_dilute['None'] += 1
            else:
                if 'hardcoded' in c:
                     cond_dilute['Top-Level Hardcoded'] += 1 # e.g. Methane
                
                # Dilute
                d = c.get('dilute', {})
                if 'hardcoded' in d:
                    cond_dilute['hardcoded'] += 1
                elif isinstance(d, dict):
                    t = d.get('type', 'Unknown')
                    cond_dilute[t] += 1
                    if t == 'Unknown':
                         print(f"{fname} Cond Dilute Unknown: {d.keys()}")
                
                # Residual
                r = c.get('residual', {})
                if r:
                    cond_residual[r.get('type', 'Unknown')] += 1
                
                # Critical
                crit = c.get('critical', {})
                if crit:
                    cond_critical[crit.get('type', 'Unknown')] += 1
                else:
                    cond_critical['None'] += 1
                    
        except Exception as e:
            print(f"Error parsing {fname}: {e}")

    print("--- Viscosity Dilute ---")
    for k, v in visc_dilute.most_common():
        print(f"{k}: {v}")
        
    print("\n--- Viscosity Initial Density ---")
    for k, v in visc_initial.most_common():
        print(f"{k}: {v}")
        
    print("\n--- Viscosity Higher Order ---")
    for k, v in visc_ho.most_common():
        print(f"{k}: {v}")
        
    print("\n--- Conductivity Dilute ---")
    for k, v in cond_dilute.most_common():
        print(f"{k}: {v}")

    print("\n--- Conductivity Residual ---")
    for k, v in cond_residual.most_common():
        print(f"{k}: {v}")
        
    print("\n--- Conductivity Critical ---")
    for k, v in cond_critical.most_common():
        print(f"{k}: {v}")

if __name__ == "__main__":
    analyze_transport()
