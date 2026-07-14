"""External tool availability checks for novelty search."""

import shutil


def require_blast():
    if shutil.which("makeblastdb") is None or shutil.which("blastn") is None:
        raise EnvironmentError(
            "BLAST+ not found on PATH (makeblastdb, blastn). "
            "Install: conda env update -f environment.yml  # adds blast from bioconda"
        )


def require_hmmer():
    if shutil.which("nhmmer") is None:
        raise EnvironmentError(
            "nhmmer not found on PATH. "
            "Install: conda env update -f environment.yml  # adds hmmer from bioconda"
        )
