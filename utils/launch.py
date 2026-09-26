"""Native Windows command-line adapter to the shared Python experiment scheduler."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", type=Path)
    mode.add_argument("--experiment")
    parser.add_argument("--layers")
    parser.add_argument("--devices", default="")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--ensemble-range", default="0:9")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-econ", action="store_true")
    parser.add_argument("--group-ablation-only", action="store_true")
    parser.add_argument("--list-experiments", action="store_true")
    args = parser.parse_args()
    from utils import tui_launcher as launcher
    from utils.runtime_device import select_devices, default_slots

    if args.list_experiments:
        for name in launcher.EXPERIMENT_ORDER:
            _, width = launcher.parse_seeded_experiment_name(name)
            print(f"{name:20} hidden layers {width}:20")
        return 0
    if args.group_ablation_only and not args.experiment:
        parser.error("--group-ablation-only requires --experiment and --layers")
    if args.experiment:
        if args.experiment not in launcher.EXPERIMENTS or not args.layers:
            parser.error("Use a listed experiment and specify --layers")
        layers = launcher.parse_integer_spec(args.layers, "layers")
        launcher.validate_layers(args.experiment, layers)
        ensembles = launcher.parse_integer_spec(args.ensemble_range, "ensemble", allow_desc=False)
        launcher.selected_ensemble_range(ensembles, experiment=args.experiment, layer=layers[0])
        devices = select_devices(args.devices, args.num_gpus)
        payload = {
            "version": 1, "devices": devices, "gpu_slots": {device: default_slots(device) if device == "mps" else 1 for device in devices},
            "ensemble_ids": ensembles, "run_econ": not args.no_econ,
            "group_ablation_only": args.group_ablation_only,
            "experiments": [{"name": args.experiment, "enabled": True, "layers": [
                {"layer": layer, "enabled": True, "cells": [
                    {"ensemble": ensemble, "enabled": True, "blocking_batch": batch,
                     "gpus": [devices[index % len(devices)]]}
                    for index, ensemble in enumerate(ensembles)
                ]} for batch, layer in enumerate(layers, start=1)
            ]}],
        }
    elif args.plan:
        payload = launcher.load_payload(args.plan.resolve())
    else:
        result = launcher.run_interactive(dry_run=args.dry_run)
        if not result:
            return 0
        payload = result["payload"]
    if args.no_econ or args.group_ablation_only:
        payload["run_econ"] = False
    launcher.execute_payload(payload, project_root=ROOT, python_bin=sys.executable, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
