from __future__ import annotations

import json
from pathlib import Path

import CoolProp.CoolProp as CP

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "chillprop" / "data"
# CoolProp exposes n-Propane in FluidsList() but the bundled JSON and existing ChillProp
# file use Propane as the canonical runtime name.
NAME_MAP = {
    "n-Propane": "Propane",
}


def existing_names() -> set[str]:
    names: set[str] = set()
    for path in DATA_DIR.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        fluid = payload[0] if isinstance(payload, list) else payload
        info = fluid.get("INFO", {})
        names.add(str(info.get("NAME", path.stem)))
    return names


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current = existing_names()
    requested = sorted({NAME_MAP.get(fluid, fluid) for fluid in CP.FluidsList()}, key=str.casefold)
    downloaded = []
    skipped = []
    for fluid in requested:
        if fluid in current:
            skipped.append(fluid)
            continue
        target = DATA_DIR / f"{fluid}.json"
        target.write_text(CP.get_fluid_param_string(fluid, "JSON"), encoding="utf-8")
        downloaded.append(fluid)
    print(json.dumps({"downloaded": downloaded, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
