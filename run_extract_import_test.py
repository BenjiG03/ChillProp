
import sys
import os
sys.path.append(os.getcwd())

print("Importing extract_params...")
from scripts.extract_params import extract_fluid_params

print("Calling extract_fluid_params('Methane')...")
try:
    data = extract_fluid_params("Methane")
    print("Transport found:", 'transport' in data)
except Exception as e:
    print("Failed:", e)
