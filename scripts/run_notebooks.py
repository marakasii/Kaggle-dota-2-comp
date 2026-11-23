"""Run the original notebooks in a shared workspace with their expected paths."""

from argparse import ArgumentParser
from pathlib import Path
import shutil
from tempfile import mkdtemp


ROOT = Path(__file__).resolve().parents[1]
CSV_FILES = (
    "matches_df_train.csv",
    "matches_df_test.csv",
    "player_df.csv",
    "Constants.Heroes.csv",
    "dota_adv.csv",
    "game_chat.csv",
)
NOTEBOOKS = (
    "homework-practice-03-base.ipynb",
    "homework-practice-03-advanced.ipynb",
)


def prepare_run():
    sources = [ROOT / "data" / name for name in CSV_FILES]
    missing = [source.name for source in sources if not source.is_file()]
    if missing:
        raise SystemExit("Missing files in data/: " + ", ".join(missing))

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    work_dir = Path(mkdtemp(prefix="run-", dir=artifacts))
    for source in sources:
        target = work_dir / source.name
        try:
            target.symlink_to(source)
        except (OSError, NotImplementedError):
            shutil.copyfile(source, target)
    return work_dir


def execute_notebooks(work_dir, kernel):
    import nbformat
    from nbclient import NotebookClient

    for name in NOTEBOOKS:
        source = ROOT / "notebooks" / name
        notebook = nbformat.read(source, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None

        print(f"Running {name}", flush=True)

        def show_progress(cell, cell_index, **kwargs):
            if cell.cell_type == "code" and cell.source.strip():
                print(f"  Cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

        client = NotebookClient(
            notebook,
            kernel_name=kernel,
            timeout=None,
            resources={"metadata": {"path": str(work_dir)}},
            on_cell_start=show_progress,
        )
        output = work_dir / name.replace(".ipynb", ".executed.ipynb")
        try:
            client.execute()
        finally:
            nbformat.write(notebook, output)
            print(f"Saved {output}", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name")
    parser.add_argument("--prepare-only", action="store_true", help="Create the workspace without training")
    args = parser.parse_args()

    work_dir = prepare_run()
    print(f"Working directory: {work_dir}", flush=True)
    if not args.prepare_only:
        execute_notebooks(work_dir, args.kernel)


if __name__ == "__main__":
    main()
