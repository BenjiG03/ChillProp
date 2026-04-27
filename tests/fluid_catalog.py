import CoolProp.CoolProp as CP

SUPPORTED_FLUIDS = [
    "Air",
    "Ammonia",
    "Argon",
    "CarbonDioxide",
    "CarbonMonoxide",
    "Cyclopentane",
    "Ethane",
    "Ethanol",
    "Ethylene",
    "HeavyWater",
    "Helium",
    "Hydrogen",
    "HydrogenSulfide",
    "IsoButane",
    "Isopentane",
    "Krypton",
    "Methane",
    "Methanol",
    "n-Butane",
    "n-Decane",
    "n-Dodecane",
    "n-Heptane",
    "n-Hexane",
    "n-Octane",
    "n-Pentane",
    "n-Undecane",
    "Neon",
    "Neopentane",
    "Nitrogen",
    "NitrousOxide",
    "Oxygen",
    "Propane",
    "Propylene",
    "R134a",
    "R32",
    "R1234yf",
    "R1234ze(E)",
    "R404A",
    "R407C",
    "R410A",
    "SulfurDioxide",
    "SulfurHexafluoride",
    "Water",
    "Xenon",
]

TRANSPORT_VALIDATED_FLUIDS = [
    "Argon",
    "Hydrogen",
    "n-Decane",
    "Nitrogen",
    "Oxygen",
    "Propane",
]

TWOPHASE_VALIDATED_FLUIDS = [
    fluid
    for fluid in SUPPORTED_FLUIDS
    if CP.get_fluid_param_string(fluid, "pure") == "true"
    and fluid not in {
        "Ethanol",
        "Hydrogen",
        "HydrogenSulfide",
        "Krypton",
        "Methanol",
        "n-Dodecane",
        "n-Heptane",
        "n-Hexane",
        "NitrousOxide",
        "Oxygen",
        "R134a",
        "R1234ze(E)",
    }
]
