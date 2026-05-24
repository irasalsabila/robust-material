import pathlib

files = [
    "src/train_baseline.py",
    "src/evaluate.py",
    "src/models/__init__.py",
    "src/models/mlp.py",
    "src/models/cnn1d.py",
    "src/utils/__init__.py",
    "src/utils/seed.py",
    "src/utils/metrics.py",
    "src/data/__init__.py",
    "src/data/xrd_dataset.py",
    "src/data/datamodule.py",
    "src/data/samplers.py",
    "src/data/transforms.py",
    "src/smoke_test_dataloader.py",
]

for fp in files:
    p = pathlib.Path(fp)
    if not p.exists():
        print(f"MISSING: {fp}")
        continue
    content = p.read_text()
    stripped = content.strip()
    # Unwrap outer triple-double-quote wrapper added by create_new_file
    if stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) > 6:
        inner = stripped[3:-3]
        p.write_text(inner)
        print(f"FIXED:   {fp}")
    else:
        print(f"OK:      {fp}")