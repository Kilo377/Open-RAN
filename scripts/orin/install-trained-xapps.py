#!/usr/bin/env python3
"""Verify the versioned six-xApp release and stage it under project outputs/."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RELEASE = "20261117"
MODELS = ("x01", "x02", "x08", "x12_mec", "x12_urllc", "x13")
FOUR_MODELS = MODELS[:4]
FORMATS = {
    "x01": "chongyu.online-energy-rf", "x02": "chongyu.online-ap-rf",
    "x08": "chongyu.online-throughput-mlp", "x12_mec": "chongyu.online-energy-rf",
    "x12_urllc": "chongyu.online-urllc-qos-rf", "x13": "chongyu.online-qgbdt",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def confined(root, relative):
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Release artifact must use a confined relative path")
    return path


def load_release(bundle):
    bundle = Path(bundle).resolve()
    data = json.loads((bundle / "release.json").read_text())
    if (data.get("schema") != "oran.chongyu-trained-model-release.v1"
            or data.get("complete") is not True or data.get("pool") != "chongyu"
            or set(data.get("models", {})) != set(MODELS)):
        raise ValueError("A complete six-model Chongyu release is required")
    files = data.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("Release file checksums are required")
    for relative, expected in files.items():
        path = confined(bundle, relative)
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError("Release checksum mismatch: " + relative)
    for name, entry in data["models"].items():
        relative = entry.get("path")
        if (relative != "models/" + name + ".json"
                or files.get(relative) != entry.get("sha256")):
            raise ValueError("Model path/checksum differs from the release: " + name)
        model = json.loads(confined(bundle, relative).read_text())
        rows, fitted = model.get("x"), model.get("trained_samples")
        if (model.get("format") != FORMATS[name] or not isinstance(rows, list)
                or not isinstance(model.get("y"), list) or len(rows) != len(model["y"])
                or type(fitted) is not int or not 0 < fitted <= len(rows)
                or type(model.get("fit_count")) is not int or model["fit_count"] < 1
                or entry.get("measured_rows") != len(rows)
                or entry.get("fitted_rows") != fitted):
            raise ValueError("Model training/schema metadata differs: " + name)
    return data


def install(bundle, output, project_root=ROOT):
    root = Path(project_root).resolve()
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if not output.is_relative_to(root / "outputs") or output == root / "outputs":
        raise ValueError("Install into a dedicated directory under project outputs/")
    data = load_release(bundle)
    content = {name: confined(bundle, name).read_bytes() for name in data["files_sha256"]}
    content["release.json"] = (bundle / "release.json").read_bytes()
    content["models.json"] = encoded({
        "schema": "oran.frozen-modelset.v1", "release": data["release"],
        "models": {name: {"path": data["models"][name]["path"],
                          "sha256": data["models"][name]["sha256"]} for name in FOUR_MODELS},
    })
    args = {}
    for name, entry in data["models"].items():
        flag = entry["cli_mode"]
        model = str((output / entry["path"]).relative_to(root))
        values = ["--" + flag + "-xapp-mode", "frozen", "--" + flag + "-xapp-model", model]
        if name == "x01":
            values += ["--energy-xapp-deploy-drx" if entry.get("runtime_parameters",{}).get("deploy_drx",False) else "--no-energy-xapp-deploy-drx"]
        if name == "x12_urllc":
            values += ["--urllc-xapp-policy-version", "qos_v2"]
        if name == "x08":
            values += ["--throughput-xapp-policy", "throughput"]
        if name == "x13":
            values += ["--budget-xapp-gflops", ".75"]
        args[name] = values
    content["runtime-arguments.json"] = encoded(args)
    if output.exists():
        if not output.is_dir() or any(not confined(output, name).is_file()
                or confined(output, name).read_bytes() != value for name, value in content.items()):
            raise ValueError("Existing install differs; use a fresh versioned output directory")
        return {"status": "already_installed", "models": len(MODELS),
                "output": str(output), "release_sha256": digest(bundle / "release.json")}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".xapp-install-", dir=output.parent))
    try:
        for relative, value in content.items():
            path = confined(temp, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        for relative, expected in data["files_sha256"].items():
            if digest(temp / relative) != expected:
                raise ValueError("Staged artifact checksum mismatch")
        if output.exists():
            raise ValueError("Destination appeared during staging; refusing overwrite")
        os.rename(temp, output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return {"status": "installed", "models": len(MODELS), "output": str(output),
            "release_sha256": digest(bundle / "release.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
        default=ROOT / "artifacts/xapps/chongyu" / RELEASE)
    parser.add_argument("--output", type=Path,
        default=ROOT / "outputs/trained_xapps" / RELEASE)
    args = parser.parse_args(argv)
    bundle = args.bundle if args.bundle.is_absolute() else ROOT / args.bundle
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(install(bundle, output), indent=2))


if __name__ == "__main__":
    main()
