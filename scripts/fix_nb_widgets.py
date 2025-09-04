# scripts/fix_nb_widgets.py
import sys, glob, nbformat, os

# Collect paths (expand any globs passed on the command line)
args = sys.argv[1:]
paths = []
if args:
    for a in args:
        paths.extend(glob.glob(a))
else:
    paths = glob.glob(os.path.join("notebooks", "*.ipynb"))

if not paths:
    print("No notebooks found.")
    raise SystemExit(0)

for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Remove broken ipywidgets metadata (root + per-cell)
    nb.metadata.pop("widgets", None)
    for c in nb.cells:
        md = c.get("metadata", {})
        if "widgets" in md:
            md.pop("widgets", None)
            c["metadata"] = md

    with open(p, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Cleaned: {p}")
