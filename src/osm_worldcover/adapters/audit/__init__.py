"""Read back release files and independently check publication guarantees.

Python keeps only one Arrow batch and bounded diagnostic samples. Global
identity checks run in DuckDB with a memory limit and disk-backed spill space.
The input files are never modified. Passing this audit does not independently
recompute raster labels, or prove that excluded source polygons were processed.
"""

from .report import AuditProblem, AuditReport
from .run import audit_build
from .schema import _SCHEMA as _SCHEMA
from .sql import _retained_text_diagnostics as _retained_text_diagnostics

__all__ = ["AuditProblem", "AuditReport", "audit_build"]

AuditProblem.__module__ = __name__
AuditReport.__module__ = __name__
