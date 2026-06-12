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


def fetch_rfam_thermoswitches(output_path="data/raw/rfam_templates.csv"):
    """Connects to the public EMBL-EBI Rfam instance to isolate RNA thermometers."""
    connection = connect()

    sql_query = """
    SELECT
        f.rfam_acc,
        f.rfam_id,
        f.description,
        f.type,
        fr.rfamseq_acc,
        fr.seq_start,
        fr.seq_end
    FROM family f
    JOIN full_region fr ON f.rfam_acc = fr.rfam_acc
    WHERE (f.description LIKE '%thermometer%'
       OR f.description LIKE '%thermoswitch%'
       OR f.type LIKE '%thermometer%')
       AND fr.is_significant = 1;
    """

    print("Executing text-mining query across Rfam relational tables...")
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
        f"with {len(df)} structural instances."
    )
    return df


if __name__ == "__main__":
    fetch_rfam_thermoswitches()
