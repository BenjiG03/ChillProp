from __future__ import annotations

import json
from pathlib import Path

import CoolProp.CoolProp as CP

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "chillprop" / "data"


def _load_supported_fluids() -> list[str]:
    fluids: list[str] = []
    for path in sorted(DATA_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        fluid = payload[0] if isinstance(payload, list) else payload
        fluids.append(str(fluid.get("INFO", {}).get("NAME", path.stem)))
    return sorted(set(fluids), key=str.casefold)


SUPPORTED_FLUIDS = _load_supported_fluids()

TRANSPORT_VALIDATED_FLUIDS = [
    "Argon",
    "Hydrogen",
    "n-Decane",
    "Nitrogen",
    "Oxygen",
    "Propane",
]

TWOPHASE_EXCLUDED_FLUIDS = {
    "CycloPropane",
    "D4",
    "D5",
    "D6",
    "Dichloroethane",
    "DiethylEther",
    "Ethanol",
    "EthyleneOxide",
    "HFE143m",
    "Hydrogen",
    "HydrogenSulfide",
    "Isohexane",
    "Krypton",
    "MD3M",
    "Methanol",
    "n-Dodecane",
    "n-Heptane",
    "n-Hexane",
    "n-Nonane",
    "NitrousOxide",
    "Oxygen",
    "ParaHydrogen",
    "Propyne",
    "R114",
    "R124",
    "R1243zf",
    "R134a",
    "R125",
    "R13",
    "R14",
    "R161",
    "R21",
    "R236EA",
    "R245fa",
    "R365MFC",
    "R1234ze(E)",
    "R40",
    "R41",
}

TWOPHASE_VALIDATED_FLUIDS = [
    fluid
    for fluid in SUPPORTED_FLUIDS
    if CP.get_fluid_param_string(fluid, "pure") == "true" and fluid not in TWOPHASE_EXCLUDED_FLUIDS
]
