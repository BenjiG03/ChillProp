import CoolProp.CoolProp as CP
import numpy as np

fluid = "Air"
T = 300
P = 101325

print(f"--- {fluid} Enthalpy at T={T}, P={P} ---")
h_def = CP.PropsSI("H", "T", T, "P", P, fluid)
print(f"Default: {h_def}")

try:
    CP.set_reference_state(fluid, "IIR")
    print(f"IIR: {CP.PropsSI('H', 'T', T, 'P', P, fluid)}")
except:
    print("IIR failed")

try:
    CP.set_reference_state(fluid, "ASHRAE")
    print(f"ASHRAE: {CP.PropsSI('H', 'T', T, 'P', P, fluid)}")
except:
    print("ASHRAE failed")

try:
    CP.set_reference_state(fluid, "NBP")
    print(f"NBP: {CP.PropsSI('H', 'T', T, 'P', P, fluid)}")
except:
    print("NBP failed")

try:
    # Reset to EOS default
    CP.set_reference_state(fluid, "RESET")
    print(f"RESET (EOS natural): {CP.PropsSI('H', 'T', T, 'P', P, fluid)}")
except:
    print("RESET failed")

fluid = "Nitrogen"
print(f"--- {fluid} Enthalpy ---")
CP.set_reference_state(fluid, "RESET")
print(f"RESET (EOS natural): {CP.PropsSI('H', 'T', T, 'P', P, fluid)}")
h_def = CP.PropsSI("H", "T", T, "P", P, fluid)
print(f"After Reset Default: {h_def}")
