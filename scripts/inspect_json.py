import sys
import json
from scripts.extract_params import extract_fluid_params

def inspect():
    data = extract_fluid_params('Nitrogen')
    if isinstance(data, list):
        data = data[0]
        
    print("Top level keys:", data.keys())
    if 'EOS' in data:
        eos = data['EOS'][0]
        print("EOS keys:", eos.keys())
        if 'alpha0' in eos:
            print("alpha0 types:", [t['type'] for t in eos['alpha0']])
        if 'alphar' in eos:
            print("alphar types:", [t['type'] for t in eos['alphar']])

if __name__ == "__main__":
    inspect()
