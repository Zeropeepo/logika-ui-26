"""Create separate source/data transfer archives without bundling secrets or weights."""
import hashlib
import json
from pathlib import Path
import zipfile


def package(inputs, output):
    source = Path(__file__).resolve().parent
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "kumo_hpc_source.zip", output / "kumo_hpc_inputs.zip"]
    if any(p.exists() for p in paths):
        raise ValueError("Transfer archives already exist; use a new output directory")
    with zipfile.ZipFile(paths[0], "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(source.rglob("*")):
            if p.is_file() and p.suffix in {".py", ".md", ".json", ".txt", ".sh"} and "__pycache__" not in p.parts:
                z.write(p, Path("HPC/kumo_suite") / p.relative_to(source))
        z.writestr(".gitignore", ".venv-kumo/\n__pycache__/\n*.pyc\ndata/kumo_inputs/\nHPC/kumo_results/\ntransfer/\n.env\n.env.*\n")
    with zipfile.ZipFile(paths[1], "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("train.parquet", "test.parquet", "folds.parquet", "manifest.json"):
            z.write(Path(inputs) / name, Path("data/kumo_inputs") / name)
    manifest = {}
    for p in paths:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        manifest[p.name] = {"bytes": p.stat().st_size, "sha256": h.hexdigest()}
    (output / "SHA256.json").write_text(json.dumps(manifest, indent=2))
    return manifest
