
import CoolProp.CoolProp as CP
import json
import sys
import os
import subprocess

def _extract_fluid_params_core(fluid_name):
    try:
        # 1. Try simple get
        try:
            fluid_json_str = CP.get_fluid_param_string(fluid_name, "JSON")
            temp_data = json.loads(fluid_json_str)
            if isinstance(temp_data, list): temp_data = temp_data[0]
            
            # Robust check for conductivity data (case-insensitive)
            keys_lower = {k.lower(): k for k in temp_data.keys()}
            t_key = keys_lower.get('transport')
            has_transport = False
            if t_key:
                trans = temp_data[t_key]
                c_keys = {k.lower(): k for k in trans.keys()}
                c_key = c_keys.get('conductivity')
                if c_key:
                    cond = trans[c_key]
                    if isinstance(cond, list): cond = cond[0]
                    if 'dilute' in {k.lower(): k for k in cond.keys()}:
                        has_transport = True
            
            if not has_transport:
                sys.stderr.write(f"DEBUG: Transport missing/incomplete for {fluid_name}. Triggering fallback.\n")
                raise ValueError("Incomplete Transport Data")
                
        except ValueError:
            # 2. Try AbstractState fallback
            backend = 'HEOS'
            try:
                asi = CP.AbstractState(backend, fluid_name)
                asi.update(CP.PT_INPUTS, 101325, 300) 
                # Force calc
                try: asi.conductivity() 
                except: pass
                
                fluid_json_str = CP.get_fluid_param_string(fluid_name, "JSON")
                
                # Check again
                check = json.loads(fluid_json_str)
                if isinstance(check, list): check = check[0]
                if 'transport' not in check:
                     # Try HEOS prefix
                     sys.stderr.write(f"DEBUG: Still missing transport. Trying HEOS::{fluid_name}\n")
                     asi = CP.AbstractState(backend, f"HEOS::{fluid_name}")
                     asi.update(CP.PT_INPUTS, 101325, 300)
                     fluid_json_str = CP.get_fluid_param_string(f"HEOS::{fluid_name}", "JSON")
            except Exception as e:
                # Fallback failed
                pass
            
        data = json.loads(fluid_json_str)
        if isinstance(data, list): data = data[0]
        return data
        
    except Exception as e:
        sys.stderr.write(f"Core extraction error for {fluid_name}: {e}\n")
        return None

def extract_fluid_params(fluid_name):
    # Try in-process first
    data = _extract_fluid_params_core(fluid_name)
    
    # Check if successful
    has_transport = False
    if data and 'transport' in data:
        cond = data['transport'].get('conductivity', {})
        if cond and 'dilute' in cond:
            has_transport = True
            
    if has_transport:
        return data
        
    # If failed, use subprocess workaround
    sys.stderr.write(f"DEBUG: In-process extraction failed for {fluid_name}. Spawning subprocess.\n")
    try:
        script_path = os.path.abspath(__file__)
        # Run this script as main
        result = subprocess.run(
            [sys.executable, script_path, fluid_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout.strip()
        # Find JSON start
        idx_brace = output.find('{')
        idx_bracket = output.find('[')
        start = 0
        if idx_brace != -1 and (idx_bracket == -1 or idx_brace < idx_bracket):
            start = idx_brace
        elif idx_bracket != -1:
            start = idx_bracket
            
        json_str = output[start:]
        final_data = json.loads(json_str)
        if isinstance(final_data, list): final_data = final_data[0]
        
        # Verify valid transport (case-insensitive)
        keys_lower = {k.lower(): k for k in final_data.keys()}
        t_key = keys_lower.get('transport')
        has_transport = False
        if t_key:
            trans = final_data[t_key]
            c_key = {k.lower(): k for k in trans.keys()}.get('conductivity')
            if c_key:
                cond = trans[c_key]
                # data.v.dilute might be a list or dict
                if isinstance(cond, list): cond = cond[0]
                if 'dilute' in {k.lower(): k for k in cond.keys()}:
                    has_transport = True
            
        if not has_transport:
             sys.stderr.write("DEBUG: Subprocess returned data without transport!\n")
             sys.stderr.write(f"Subprocess STDERR:\n{result.stderr}\n")
             
        return final_data
        
    except Exception as e:
        sys.stderr.write(f"Subprocess extraction failed: {e}\n")
        return data # Return partial data if subprocess fails

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_params.py <FluidName> [OutputFile]")
        sys.exit(1)
        
    fluid = sys.argv[1]
    # Call CORE directly when running as script to avoid infinite recursion loop
    # (Though logic prevents it, this is cleaner)
    data = _extract_fluid_params_core(fluid)
    
    if data:
        if len(sys.argv) >= 3:
            with open(sys.argv[2], 'w') as f:
                json.dump(data[0] if isinstance(data, list) else data, f, indent=2)
            print(f"Parameters for {fluid} written to {sys.argv[2]}")
        else:
            print(json.dumps(data[0] if isinstance(data, list) else data, indent=2))
    else:
        sys.exit(1)
