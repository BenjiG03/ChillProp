
import subprocess
import os
import sys
import json

fluids = [
    "Nitrogen", "Oxygen", "Argon", "Air", 
    "Hydrogen", "CarbonDioxide", "Water",
    "Methane", "Ethane", "Propane", 
    "n-Butane", "IsoButane", "n-Dodecane"
]

data_dir = "chillprop/data"
os.makedirs(data_dir, exist_ok=True)

script_path = "scripts/extract_params.py"

for f in fluids:
    print(f"Proc-Extracting {f}...")
    try:
        # Run the script as a fresh process
        result = subprocess.run(
            [sys.executable, script_path, f],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        
        output = result.stdout.strip()
        # Find JSON start (to skip any debug prints)
        idx_brace = output.find('{')
        idx_bracket = output.find('[')
        start = 0
        if idx_brace != -1 and (idx_bracket == -1 or idx_brace < idx_bracket):
            start = idx_brace
        elif idx_bracket != -1:
            start = idx_bracket
            
        json_str = output[start:]
        data = json.loads(json_str)
        if isinstance(data, list): data = data[0]
        
        # Verify transport
        if 'transport' not in data:
            print(f"!!! {f} STILL MISSING TRANSPORT !!!")
        else:
            print(f"Success: {f} transport found.")
            
        # Save
        with open(os.path.join(data_dir, f"{f}.json"), 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=2)
            
    except Exception as e:
        print(f"Error extracting {f}: {e}")
        if hasattr(e, 'stderr'):
            print(f"Stderr: {e.stderr}")
