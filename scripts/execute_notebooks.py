"""Execute the notebooks in place so plots and tables are stored in the .ipynb.

    python scripts/execute_notebooks.py            # all notebooks
    python scripts/execute_notebooks.py 01_EDA     # one notebook (name prefix)
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def main(prefixes: list[str]) -> None:
    paths = sorted(NB_DIR.glob("*.ipynb"))
    if prefixes:
        paths = [p for p in paths if any(p.name.startswith(x) for x in prefixes)]
    for path in paths:
        t0 = time.time()
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(NB_DIR)}}).execute()
        nbformat.write(nb, path)
        n = sum(1 for c in nb.cells if c.cell_type == "code")
        print(f"executed {path.name}: {n} code cells in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
