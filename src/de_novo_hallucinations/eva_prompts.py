"""EVA Option B panel: host → TaxID + quota (no hardcoded Greengenes strings).

EVA's official CLI/Docker wrapper expands --taxid into the Greengenes lineage
before calling the model. RNA type must be mRNA (thermoswitches are 5′ UTR cis
elements), never sRNA.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# Confirm "mRNA" spelling against EVA Condition Control at deploy time.
DEFAULT_RNA_TYPE = "mRNA"


@dataclass(frozen=True)
class PanelHost:
    name: str
    taxid: int
    n_seqs: int


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable, falling back to default."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def load_panel_hosts(*, smoke: bool = False) -> list[PanelHost]:
    """Return Option B host panel.

    Smoke mode uses tiny per-host quotas that still exercise all three TaxIDs
    unless EVA_SMOKE_SINGLE_HOST=1 (E. coli only).
    """
    rna_type = os.environ.get("EVA_RNA_TYPE", DEFAULT_RNA_TYPE).strip()
    if rna_type.lower() in {"srna", "s_rna", "small_rna"}:
        raise ValueError(
            f"EVA_RNA_TYPE={rna_type!r} is invalid for thermoswitch design; use mRNA"
        )

    hosts = [
        PanelHost(
            name="ecoli",
            taxid=_env_int("EVA_PANEL_ECOLI_TAXID", 562),
            n_seqs=_env_int("EVA_PANEL_ECOLI_N", 3334),
        ),
        PanelHost(
            name="salmonella",
            taxid=_env_int("EVA_PANEL_SALMONELLA_TAXID", 28901),
            n_seqs=_env_int("EVA_PANEL_SALMONELLA_N", 3333),
        ),
        PanelHost(
            name="listeria",
            taxid=_env_int("EVA_PANEL_LISTERIA_TAXID", 1639),
            n_seqs=_env_int("EVA_PANEL_LISTERIA_N", 3333),
        ),
    ]

    if smoke:
        single = os.environ.get("EVA_SMOKE_SINGLE_HOST", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        smoke_total = _env_int("EVA_NUM_SAMPLES", 16)
        if single:
            return [
                PanelHost(name=hosts[0].name, taxid=hosts[0].taxid, n_seqs=smoke_total)
            ]
        # Distribute tiny smoke quota across three hosts (at least 1 each if possible).
        per = max(smoke_total // 3, 1)
        remainder = smoke_total - per * 3
        out = []
        for index, host in enumerate(hosts):
            n = per + (1 if index < remainder else 0)
            if n > 0:
                out.append(PanelHost(name=host.name, taxid=host.taxid, n_seqs=n))
        return out

    return hosts


def rna_type() -> str:
    """Return the EVA RNA type, rejecting sRNA values."""
    value = os.environ.get("EVA_RNA_TYPE", DEFAULT_RNA_TYPE).strip() or DEFAULT_RNA_TYPE
    if value.lower() in {"srna", "s_rna", "small_rna"}:
        raise ValueError("Do not use sRNA for EVA thermoswitch generation; use mRNA")
    return value


def panel_total(hosts: list[PanelHost] | None = None) -> int:
    """Return the total sequence quota across the host panel."""
    hosts = hosts if hosts is not None else load_panel_hosts(smoke=False)
    return sum(h.n_seqs for h in hosts)
