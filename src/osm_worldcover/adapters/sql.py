"""SQL text helpers shared by the DuckDB-backed adapters."""


def sql_literal(value: str) -> str:
    """Quote ``value`` as a SQL string literal, so a quote in it cannot end the literal early."""
    return "'" + value.replace("'", "''") + "'"
