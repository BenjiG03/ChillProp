import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import CoolProp.CoolProp as CP
import chillprop.highlevel as CH
from fluid_catalog import SUPPORTED_FLUIDS, TRANSPORT_VALIDATED_FLUIDS, TWOPHASE_VALIDATED_FLUIDS
from test_highlevel_pure_jax import (
    CORE_RTOL,
    CORE_SINGLE_PHASE_OUTPUTS,
    OUTPUT_RTOL,
    TRANSPORT_OUTPUTS,
    TRIVIAL_KEYS,
    TRIVIAL_RTOL,
    TWOPHASE_OUTPUTS,
    TWOPHASE_RTOL,
    _single_phase_grid,
    _twophase_grid,
)

PLOTS_DIR = ROOT / "docs" / "plots" / "validated"
STATS_PATH = ROOT / "docs" / "wiki" / "validation_stats.json"
SPHINX_VALIDATION = ROOT / "docs" / "validation.rst"
WIKI_VALIDATION = ROOT / "docs" / "wiki" / "Validation.md"

PLOT_OUTPUTS = [
    ("Dmolar", "Molar Density"),
    ("Hmolar", "Molar Enthalpy"),
    ("Smolar", "Molar Entropy"),
    ("A", "Speed of Sound"),
]


def rel_err(actual: float, expected: float) -> float:
    denom = max(abs(expected), 1e-12)
    return abs(actual - expected) / denom


def coolprop_trivial(key: str, fluid: str) -> float:
    if key == "Pmax":
        state = CP.AbstractState("HEOS", fluid)
        return float(state.keyed_output(CP.iP_max))
    if key == "Pmin":
        state = CP.AbstractState("HEOS", fluid)
        return float(state.keyed_output(CP.iP_min))
    return float(CP.PropsSI(key, fluid))


def summarize_metric(metrics: dict[str, dict[str, float]], tolerance: float, num_state_points: int) -> dict[str, float]:
    all_errs = [entry["max_rel_err"] for entry in metrics.values()]
    mean_errs = [entry["mean_rel_err"] for entry in metrics.values()]
    return {
        "num_state_points": num_state_points,
        "num_output_checks": num_state_points * len(metrics),
        "max_rel_err": max(all_errs) if all_errs else None,
        "mean_rel_err": float(np.mean(mean_errs)) if mean_errs else None,
        "tolerance_default": tolerance,
    }


def note_for_fluid(fluid: str) -> list[str]:
    notes: list[str] = []
    if fluid not in TRANSPORT_VALIDATED_FLUIDS:
        notes.append(
            "Transport properties may be implemented, but this fluid is outside the automated transport parity subset."
        )
    if CP.get_fluid_param_string(fluid, "pure") != "true":
        notes.append("CoolProp does not report this fluid as pure, so the automated two-phase grid is not applied.")
    elif fluid not in TWOPHASE_VALIDATED_FLUIDS:
        notes.append(
            "This fluid is outside the documented automated two-phase subset because its current saturation parity exceeds the published tolerance."
        )
    return notes


def compute_stats() -> dict:
    stats: dict[str, object] = {
        "supported_fluids": SUPPORTED_FLUIDS,
        "transport_validated_fluids": TRANSPORT_VALIDATED_FLUIDS,
        "twophase_validated_fluids": TWOPHASE_VALIDATED_FLUIDS,
        "fluids": {},
    }
    fluids_stats: dict[str, dict] = {}
    for fluid in SUPPORTED_FLUIDS:
        fluid_stats: dict[str, object] = {}

        trivial_stats = {}
        trivial_errs = []
        for key in TRIVIAL_KEYS:
            actual = float(CH.PropsSI(key, fluid))
            expected = coolprop_trivial(key, fluid)
            error = rel_err(actual, expected)
            trivial_stats[key] = {"actual": actual, "expected": expected, "rel_err": error}
            trivial_errs.append(error)
        fluid_stats["trivial"] = trivial_stats
        fluid_stats["trivial_summary"] = {
            "num_checks": len(TRIVIAL_KEYS),
            "max_rel_err": max(trivial_errs),
            "mean_rel_err": float(np.mean(trivial_errs)),
            "tolerance": TRIVIAL_RTOL,
        }

        single_phase = {}
        T_sp, P_sp = _single_phase_grid(fluid)
        for output in CORE_SINGLE_PHASE_OUTPUTS:
            errs = []
            for t, p in zip(T_sp, P_sp):
                actual = float(CH.PropsSI(output, "T", float(t), "P", float(p), fluid))
                expected = float(CP.PropsSI(output, "T", float(t), "P", float(p), fluid))
                errs.append(rel_err(actual, expected))
            single_phase[output] = {
                "num_checks": len(errs),
                "max_rel_err": max(errs),
                "mean_rel_err": float(np.mean(errs)),
                "tolerance": OUTPUT_RTOL.get(output, CORE_RTOL),
            }
        fluid_stats["single_phase"] = single_phase
        fluid_stats["single_phase_summary"] = summarize_metric(single_phase, CORE_RTOL, len(T_sp))

        transport = {}
        if fluid in TRANSPORT_VALIDATED_FLUIDS:
            for output in TRANSPORT_OUTPUTS:
                errs = []
                for t, p in zip(T_sp, P_sp):
                    actual = float(CH.PropsSI(output, "T", float(t), "P", float(p), fluid))
                    expected = float(CP.PropsSI(output, "T", float(t), "P", float(p), fluid))
                    errs.append(rel_err(actual, expected))
                transport[output] = {
                    "num_checks": len(errs),
                    "max_rel_err": max(errs),
                    "mean_rel_err": float(np.mean(errs)),
                    "tolerance": OUTPUT_RTOL[output],
                }
            fluid_stats["transport_summary"] = summarize_metric(transport, OUTPUT_RTOL["V"], len(T_sp))
        fluid_stats["transport"] = transport

        twophase = {}
        if fluid in TWOPHASE_VALIDATED_FLUIDS:
            T_tp, Q_tp = _twophase_grid(fluid)
            for output in TWOPHASE_OUTPUTS:
                errs = []
                for t, q in zip(T_tp, Q_tp):
                    actual = float(CH.PropsSI(output, "T", float(t), "Q", float(q), fluid))
                    expected = float(CP.PropsSI(output, "T", float(t), "Q", float(q), fluid))
                    errs.append(rel_err(actual, expected))
                twophase[output] = {
                    "num_checks": len(errs),
                    "max_rel_err": max(errs),
                    "mean_rel_err": float(np.mean(errs)),
                    "tolerance": OUTPUT_RTOL.get(output, TWOPHASE_RTOL),
                }
            fluid_stats["two_phase_summary"] = summarize_metric(twophase, TWOPHASE_RTOL, len(T_tp))
        fluid_stats["two_phase"] = twophase

        fluid_stats["notes"] = note_for_fluid(fluid)
        fluids_stats[fluid] = fluid_stats

    stats["fluids"] = fluids_stats
    return stats


def format_err(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3e}"


def generate_plot(fluid: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    T_vals, P_vals = _single_phase_grid(fluid)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.ravel()
    for ax, (output, label) in zip(axes, PLOT_OUTPUTS):
        expected = np.array(
            [float(CP.PropsSI(output, "T", float(t), "P", float(p), fluid)) for t, p in zip(T_vals, P_vals)],
            dtype=float,
        )
        actual = np.array(
            [float(CH.PropsSI(output, "T", float(t), "P", float(p), fluid)) for t, p in zip(T_vals, P_vals)],
            dtype=float,
        )
        max_abs = max(np.max(np.abs(expected)), np.max(np.abs(actual)), 1.0)
        lo = min(np.min(expected), np.min(actual))
        hi = max(np.max(expected), np.max(actual))
        if math.isclose(lo, hi):
            lo -= 1.0
            hi += 1.0
        ax.scatter(expected, actual, color="#0b6e4f", s=28)
        ax.plot([lo, hi], [lo, hi], color="#b22222", linewidth=1.2)
        for x, y, t, p in zip(expected, actual, T_vals, P_vals):
            ax.annotate(f"{t:.0f} K\n{p/1e6:.1f} MPa", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_title(label)
        ax.set_xlabel("CoolProp")
        ax.set_ylabel("ChillProp")
        if max_abs > 1e3:
            ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        errs = [rel_err(a, e) for a, e in zip(actual, expected)]
        ax.text(0.04, 0.96, f"max rel err {max(errs):.2e}", transform=ax.transAxes, va="top", fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(f"{fluid} Single-Phase Parity Grid")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{fluid}_parity.png", dpi=160)
    plt.close(fig)


def build_summary_rows(stats: dict) -> list[list[str]]:
    rows = []
    for fluid in SUPPORTED_FLUIDS:
        fluid_stats = stats["fluids"][fluid]
        rows.append(
            [
                f"``{fluid}``",
                f"``{format_err(fluid_stats['trivial_summary']['max_rel_err'])}``",
                f"``{format_err(fluid_stats['single_phase_summary']['max_rel_err'])}``",
                f"``{format_err(fluid_stats.get('transport_summary', {}).get('max_rel_err'))}``",
                f"``{format_err(fluid_stats.get('two_phase_summary', {}).get('max_rel_err'))}``",
                "; ".join(fluid_stats["notes"]) if fluid_stats["notes"] else "Full documented suite for current subsets",
            ]
        )
    return rows


def write_sphinx_validation(stats: dict) -> None:
    rows = build_summary_rows(stats)
    lines = [
        "Validation",
        "==========",
        "",
        "This page summarizes the current automated parity coverage exercised against CoolProp for the documented ChillProp fluid catalog. The statistics and plot gallery on this page are derived from the same grids used by ``tests/test_highlevel_pure_jax.py``.",
        "",
        "Catalog Snapshot",
        "----------------",
        "",
        f"* Supported runtime fluids: ``{len(SUPPORTED_FLUIDS)}``",
        f"* Automated transport subset: ``{len(TRANSPORT_VALIDATED_FLUIDS)}`` fluids",
        f"* Automated two-phase subset: ``{len(TWOPHASE_VALIDATED_FLUIDS)}`` fluids",
        "",
        "Supported fluids:",
        "",
        "* " + ", ".join(f"``{fluid}``" for fluid in SUPPORTED_FLUIDS),
        "",
        "Automated transport subset:",
        "",
        "* " + ", ".join(f"``{fluid}``" for fluid in TRANSPORT_VALIDATED_FLUIDS),
        "",
        "Automated two-phase subset:",
        "",
        "* " + ", ".join(f"``{fluid}``" for fluid in TWOPHASE_VALIDATED_FLUIDS),
        "",
        "Validation Scope",
        "----------------",
        "",
        "* Trivial fluid constants: 11 checks per fluid.",
        "* Single-phase core parity grid: 4 state points by 20 outputs per supported fluid.",
        "* Transport parity grid: 4 state points by 3 outputs for the dedicated transport subset.",
        "* Two-phase parity grid: 6 state points by 10 outputs for the documented two-phase subset.",
        "",
        "Tolerances",
        "----------",
        "",
        "+----------------------------+-------------------+-----------------------------------------------------------------------+",
        "| Category                   | Default tolerance | Notes                                                                 |",
        "+============================+===================+=======================================================================+",
        "| Trivial outputs            | ``3e-2``          | Scalar constants such as critical properties and molar mass           |",
        "+----------------------------+-------------------+-----------------------------------------------------------------------+",
        "| Single-phase core outputs  | ``5e-8``          | Property-specific relaxations are applied in the test suite           |",
        "+----------------------------+-------------------+-----------------------------------------------------------------------+",
        "| Two-phase outputs          | ``1e-3``          | ``Q`` uses ``1e-4``; only the documented subset is enforced           |",
        "+----------------------------+-------------------+-----------------------------------------------------------------------+",
        "| Transport outputs          | ``5e-3``          | Viscosity, conductivity, and Prandtl number for the transport subset  |",
        "+----------------------------+-------------------+-----------------------------------------------------------------------+",
        "",
        "Supported Materials Summary",
        "---------------------------",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 16 14 16 16 16 32",
        "",
        "   * - Fluid",
        "     - Trivial max rel err",
        "     - Single-phase max rel err",
        "     - Transport max rel err",
        "     - Two-phase max rel err",
        "     - Notes",
    ]
    for row in rows:
        lines.extend(
            [
                f"   * - {row[0]}",
                f"     - {row[1]}",
                f"     - {row[2]}",
                f"     - {row[3]}",
                f"     - {row[4]}",
                f"     - {row[5]}",
            ]
        )
    lines.extend(
        [
            "",
            "Plot Gallery",
            "------------",
            "",
            "The complete per-fluid single-phase parity plot set is stored in ``docs/plots/validated`` and indexed below.",
            "",
        ]
    )
    for fluid in SUPPORTED_FLUIDS:
        lines.extend(
            [
                fluid,
                "^" * len(fluid),
                "",
                f".. image:: plots/validated/{fluid}_parity.png",
                f"   :alt: {fluid} parity plot",
                "",
            ]
        )
    SPHINX_VALIDATION.write_text("\n".join(lines), encoding="utf-8")


def write_wiki_validation(stats: dict) -> None:
    rows = build_summary_rows(stats)
    lines = [
        "# Validation",
        "",
        "This page summarizes the current automated parity coverage exercised against CoolProp for the documented ChillProp fluid catalog. The statistics and plot gallery on this page are derived from the same grids used by `tests/test_highlevel_pure_jax.py`.",
        "",
        "## Catalog Snapshot",
        "",
        f"- Supported runtime fluids: `{len(SUPPORTED_FLUIDS)}`",
        f"- Automated transport subset: `{len(TRANSPORT_VALIDATED_FLUIDS)}` fluids",
        f"- Automated two-phase subset: `{len(TWOPHASE_VALIDATED_FLUIDS)}` fluids",
        f"- Supported fluids: {', '.join(f'`{fluid}`' for fluid in SUPPORTED_FLUIDS)}",
        f"- Transport subset: {', '.join(f'`{fluid}`' for fluid in TRANSPORT_VALIDATED_FLUIDS)}",
        f"- Two-phase subset: {', '.join(f'`{fluid}`' for fluid in TWOPHASE_VALIDATED_FLUIDS)}",
        "",
        "## Validation Scope",
        "",
        "- Trivial fluid constants: 11 checks per fluid",
        "- Single-phase core parity grid: 4 state points x 20 outputs per supported fluid",
        "- Transport parity grid: 4 state points x 3 outputs for the dedicated transport subset",
        "- Two-phase parity grid: 6 state points x 10 outputs for the documented two-phase subset",
        "",
        "## Tolerances",
        "",
        "| Category | Default tolerance | Notes |",
        "| :--- | ---: | :--- |",
        "| Trivial outputs | `3e-2` | Scalar constants such as critical properties and molar mass |",
        "| Single-phase core outputs | `5e-8` | Property-specific relaxations are applied in the test suite |",
        "| Two-phase outputs | `1e-3` | `Q` uses `1e-4`; only the documented subset is enforced |",
        "| Transport outputs | `5e-3` | Viscosity, conductivity, and Prandtl number for the transport subset |",
        "",
        "## Supported Materials Summary",
        "",
        "| Fluid | Trivial max rel err | Single-phase max rel err | Transport max rel err | Two-phase max rel err | Notes |",
        "| :--- | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |")
    lines.extend(
        [
            "",
            "## Plot Gallery",
            "",
            "The complete per-fluid single-phase parity plot set is stored in `docs/plots/validated` and indexed below.",
            "",
        ]
    )
    for fluid in SUPPORTED_FLUIDS:
        lines.extend([f"### `{fluid}`", "", f"![{fluid} parity plot](../plots/validated/{fluid}_parity.png)", ""])
    WIKI_VALIDATION.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    stats = compute_stats()
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    for fluid in SUPPORTED_FLUIDS:
        generate_plot(fluid)
    write_sphinx_validation(stats)
    write_wiki_validation(stats)


if __name__ == "__main__":
    main()
