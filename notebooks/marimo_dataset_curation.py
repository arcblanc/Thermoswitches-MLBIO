"""Marimo App 1 moved to ``notebooks/01_rfam_refseq_curation_eda.py``.

Kept as a thin pointer so older README / script links still resolve.

Run:
    uv run marimo edit notebooks/01_rfam_refseq_curation_eda.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="App 1 moved → 01_rfam_refseq_curation_eda",
)


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
    # App 1 relocated

    Open the full Rfam vs RefSeq curation EDA notebook:

    ```bash
    uv run marimo edit notebooks/01_rfam_refseq_curation_eda.py
    ```

    Helpers: `src/thermo_sim/curation_eda.py`.
    """
    )
    return


if __name__ == "__main__":
    app.run()
