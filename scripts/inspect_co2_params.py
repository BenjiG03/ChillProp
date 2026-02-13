import sys
import os
import json
sys.path.append(os.getcwd())
from scripts.extract_params import extract_fluid_params

data = extract_fluid_params("CarbonDioxide")
print(f"Type of data: {type(data)}")
if isinstance(data, list):
    print(f"List length: {len(data)}")
    data = data[0] # Try taking first element

cond = data.get('TRANSPORT', {}).get('conductivity', {})
print(json.dumps(cond, indent=2))
