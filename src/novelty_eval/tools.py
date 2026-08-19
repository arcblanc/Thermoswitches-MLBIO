"""External tool availability checks for novelty search."""

import shutil


def require_blast() -> None:
    """Raise if BLAST+ binaries are missing from PATH."""
    if shutil.which("makeblastdb") is None or shutil.which("blastn") is None:
        raise EnvironmentError(
            "BLAST+ not found on PATH (makeblastdb, blastn). "
            "Install: conda env update -f environment.yml  # adds blast from bioconda"
        )


def require_hmmer() -> None:
    """Raise if nhmmer is missing from PATH."""
    if shutil.which("nhmmer") is None:
        raise EnvironmentError(
            "nhmmer not found on PATH. "
            "Install: conda env update -f environment.yml  # adds hmmer from bioconda"
        )
