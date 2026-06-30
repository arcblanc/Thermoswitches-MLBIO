import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import mysql.connector
import pandas as pd

DEDUP_COLUMNS = ["rfam_acc", "rfamseq_acc", "seq_start", "seq_end"]

RFAM_CONFIG = {
    "host": "mysql-rfam-public.ebi.ac.uk",
    "user": "rfamro",
    "password": "",
    "port": 4497,
    "database": "Rfam",
}


def connect():
    """Open a read-only connection to the public Rfam MySQL instance."""
    return mysql.connector.connect(**RFAM_CONFIG)


def fetch_rfam_thermoswitches(output_path="data/raw/rfam_positives.csv"):
    """Connects to the public EMBL-EBI Rfam instance to isolate heat-dependent, prokaryotic RNA thermometers."""
    connection = connect()

    sql_query = """
    SELECT
        f.rfam_acc,
        f.rfam_id,
        f.description,
        f.type,
        fr.rfamseq_acc,
        fr.seq_start,
        fr.seq_end,
        t.tax_string
    FROM family f
    JOIN full_region fr ON f.rfam_acc = fr.rfam_acc
    JOIN rfamseq rs ON fr.rfamseq_acc = rs.rfamseq_acc
    JOIN taxonomy t ON rs.ncbi_id = t.ncbi_id
    WHERE (
       -- 1. Standard Rfam Classifications (Heat-dependent only)
       f.type LIKE '%thermoregulator%'
       OR f.type LIKE '%thermometer%'
       OR f.description LIKE '%thermoregulator%'
       OR f.description LIKE '%thermometer%'
       OR f.description LIKE '%thermoswitch%'
       OR f.description LIKE '%heat shock%'
       OR f.description LIKE '%heat-shock%'

       -- 2. Canonical Heat Shock & Virulence Thermometers
       OR f.rfam_id LIKE '%ROSE%'
       OR f.rfam_id LIKE '%FourU%'
       OR f.rfam_id LIKE '%ToxT%'
       OR f.rfam_id LIKE '%AilA%'
       OR f.rfam_id LIKE '%cIII%'
       OR f.rfam_id LIKE '%agsA%'

       -- 3. Newly Identified Pathogen Thermoswitches
       OR f.rfam_id LIKE '%groES%'
       OR f.rfam_id LIKE '%clpB%'
       OR f.rfam_id LIKE '%shuA%'
       OR f.rfam_id LIKE '%chuA%'
       OR f.rfam_id LIKE '%cnfY%'
       OR f.rfam_id LIKE '%cnf1%'
       
       -- 4. Pseudomonas elements
       OR f.rfam_id LIKE '%ibpA%'
       OR f.rfam_id LIKE '%IpbA%'
       OR f.rfam_id LIKE '%rhlA%'
       OR f.rfam_id LIKE '%lasI%'
       OR f.rfam_id LIKE '%lsdI%'
       
       -- 5. Yersinia, coupled, & synthetic targets:
       OR f.rfam_id LIKE '%katA%'
       OR f.rfam_id LIKE '%cysK-2%'
       OR f.rfam_id LIKE '%sodB%'
       OR f.rfam_id LIKE '%pepN%'
       OR f.rfam_id LIKE '%trxA%'
       OR f.rfam_id LIKE '%hsp17%'
       OR f.rfam_id LIKE '%sctS%'
       OR f.rfam_id LIKE '%sctT%'
       OR f.rfam_id LIKE '%Cyanobacterial%'
    )
    AND fr.is_significant = 1
    AND t.tax_string NOT LIKE '%Eukaryota%'; -- Ensures mathematically consistent SD-occlusion models
    """

    print("Executing text-mining query across Rfam relational tables...")
    df = pd.read_sql(sql_query, connection)

    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_COLUMNS)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate row(s) before saving.")

    df.to_csv(output_path, index=False)
    legacy_path = "data/raw/rfam_templates.csv"
    if output_path != legacy_path:
        df.to_csv(legacy_path, index=False)
    connection.close()
    print(
        f"Data ingestion successful! Overwrote {output_path} "
        f"with {len(df)} structural instances."
    )
    if output_path != legacy_path:
        print(f"Also wrote legacy copy to {legacy_path}.")
    return df


def fetch_rfam_negative_controls(output_path="data/raw/rfam_negatives.csv"):
    """
    Connects to the public EMBL-EBI Rfam instance to isolate standard,
    non-thermoswitch bacterial 5' UTRs and Cis-regulatory elements as negative controls.
    """
    connection = connect()

    sql_query = """
    SELECT
        f.rfam_acc,
        f.rfam_id,
        f.description,
        f.type,
        fr.rfamseq_acc,
        fr.seq_start,
        fr.seq_end,
        t.tax_string
    FROM family f
    JOIN full_region fr ON f.rfam_acc = fr.rfam_acc
    JOIN rfamseq rs ON fr.rfamseq_acc = rs.rfamseq_acc
    JOIN taxonomy t ON rs.ncbi_id = t.ncbi_id
    WHERE (
       -- 1. Target other structural 5' UTRs, riboswitches, and leader sequences
       (f.type LIKE '%Cis-reg%'
        OR f.description LIKE '%5%UTR%'
        OR f.description LIKE '%leader%')

       -- 2. STRICT EXCLUSION: Remove any element related to temperature
       AND f.type NOT LIKE '%thermoregulator%'
       AND f.type NOT LIKE '%thermometer%'
       AND f.description NOT LIKE '%thermoregulator%'
       AND f.description NOT LIKE '%thermometer%'
       AND f.description NOT LIKE '%thermoswitch%'
       AND f.description NOT LIKE '%heat shock%'
       AND f.description NOT LIKE '%cold shock%'
       AND f.description NOT LIKE '%temperature%'
    )
    AND fr.is_significant = 1

    -- 3. Strict Taxonomic Filtering to match your positive dataset
    AND t.tax_string LIKE '%Bacteria%'
    AND t.tax_string NOT LIKE '%Eukaryota%';
    """

    print("Executing text-mining query for negative controls...")
    df = pd.read_sql(sql_query, connection)

    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_COLUMNS)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate row(s) before saving.")

    df.to_csv(output_path, index=False)
    connection.close()
    print(
        f"Data ingestion successful! Overwrote {output_path} "
        f"with {len(df)} negative structural instances."
    )
    return df


if __name__ == "__main__":
    fetch_rfam_thermoswitches()
    fetch_rfam_negative_controls()
