"""Run analytical SQL against the local SQLite orders table."""

from __future__ import annotations

import json
import sqlite3

from src.config import ARTIFACTS, SQLITE_PATH, SQL_DIR


def main() -> None:
    sql_path = SQL_DIR / "analysis.sql"
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    statements = [s.strip() for s in sql_path.read_text(encoding="utf-8").split(";") if s.strip()]
    results = []
    for stmt in statements:
        if stmt.startswith("--") and "SELECT" not in stmt.upper():
            continue
        title = "query"
        lines = stmt.splitlines()
        if lines and lines[0].strip().startswith("--"):
            title = lines[0].strip().lstrip("- ").strip()
        cur = conn.execute(stmt)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(r) for r in cur.fetchall()]
        results.append({"title": title, "columns": cols, "rows": rows, "sql": stmt})
    conn.close()
    with open(ARTIFACTS / "sql_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Ran {len(results)} SQL statements")


if __name__ == "__main__":
    main()
