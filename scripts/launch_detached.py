#!/usr/bin/env python3
"""Launch a long job fully detached from the controlling terminal (survives sleep/close).

Usage:
  python scripts/launch_detached.py --log data/processed/mac_thermo_full.log -- \\
    bash scripts/mac_thermo_resume.sh
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Detach a long-running command.")
    parser.add_argument("--log", required=True, help="Append stdout/stderr to this file")
    parser.add_argument("--pid-file", default=None, help="Write child PID here")
    parser.add_argument("--cwd", default=str(ROOT))
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("Provide a command after --")

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pid_path = Path(args.pid_file) if args.pid_file else log_path.with_suffix(".pid")
    if not pid_path.is_absolute():
        pid_path = ROOT / pid_path

    log_handle = open(log_path, "a")
    proc = subprocess.Popen(
        command,
        cwd=args.cwd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_path.write_text(str(proc.pid))
    print(f"Detached PID={proc.pid}")
    print(f"  log: {log_path}")
    print(f"  pid: {pid_path}")


if __name__ == "__main__":
    main()
