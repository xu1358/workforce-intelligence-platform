"""Create least-privilege PostgreSQL views for Version 2 analytics."""

from __future__ import annotations

from pathlib import Path

import psycopg2

from v2_data_access import (
    PROJECT_ROOT,
    TABLE_SPECS,
    load_database_config,
    load_source_config,
)


def create_views(project_root: Path = PROJECT_ROOT) -> None:
    """Create and verify every committed Version 2 source view."""

    config = load_source_config(
        project_root / "config" / "v2_postgresql_source.yaml"
    )
    sql_path = project_root / config["database"]["view_sql"]
    statements = sql_path.read_text(encoding="utf-8")
    connection = psycopg2.connect(
        **load_database_config(project_root, config)
    )

    try:
        with connection.cursor() as cursor:
            cursor.execute(statements)
            for spec in TABLE_SPECS.values():
                cursor.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = %s
                      AND table_name = %s
                    ORDER BY ordinal_position;
                    """,
                    (config["database"]["schema"], spec.relation),
                )
                actual = [row[0] for row in cursor.fetchall()]
                if actual != list(spec.columns):
                    raise ValueError(
                        f"{spec.relation} has unexpected columns. "
                        f"Expected {list(spec.columns)}; received {actual}."
                    )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    print("\nVERSION 2 POSTGRESQL ANALYTICAL VIEWS")
    for spec in TABLE_SPECS.values():
        print(
            f"PASS  {spec.relation:<44} "
            f"{len(spec.columns)} least-privilege columns"
        )
    print("\nVERSION 2 POSTGRESQL VIEWS CREATED SUCCESSFULLY")


if __name__ == "__main__":
    create_views()
