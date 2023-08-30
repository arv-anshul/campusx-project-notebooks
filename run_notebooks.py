"""Run notebooks in sequence."""

import os
from pathlib import Path
from time import perf_counter
from typing import Callable

import nbformat
from nbconvert.preprocessors.execute import ExecutePreprocessor

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
root = Path()


def get_execution_time(func: Callable, **kw):
    start = perf_counter()
    func(**kw)
    return perf_counter() - start


def read_notebook(fp: Path):
    with open(fp) as f:
        return nbformat.read(f, nbformat.NO_CONVERT)


def run_notebook(fp: Path, **kw):
    nb_in = read_notebook(fp)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3", **kw)
    ep.preprocess(nb_in)


def main():
    notebook_fp_list: list[Path] = [
        root / "new_notebooks/filter_dataset.ipynb",
        root / "new_notebooks/_OVERVIEW.ipynb",
        root / "new_notebooks/1.0_PREPROCESSING.ipynb",
        root / "new_notebooks/1.1_PREPROCESSING.ipynb",
    ]

    for fp in notebook_fp_list:
        print("--" * 40)
        print(f"{'Running:':<20} {fp}")
        exec_time = get_execution_time(run_notebook, fp=fp)
        print(f"{'Time:':<20} {round(exec_time, 3)}s")
        print(f"{'Running Complete:':<20} {fp}")
        print("--" * 40)


if __name__ == "__main__":
    main()
