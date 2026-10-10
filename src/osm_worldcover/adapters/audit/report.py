from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AuditProblem:
    """A failed guarantee, with at most five diagnostic examples."""

    code: str
    count: int
    examples: tuple[str, ...] = ()


@dataclass(slots=True)
class AuditReport:
    """All detected problems and separately identified modelling risks."""

    rows: int = 0
    problems: list[AuditProblem] = field(default_factory=list)
    warnings: list[AuditProblem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every required check passed."""
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""
        return {"ok": self.ok, **asdict(self)}


class _Checks:
    def __init__(self, report: AuditReport):
        self.report = report
        self.counts: Counter[str] = Counter()
        self.samples: defaultdict[str, list[str]] = defaultdict(list)

    def add(self, code: str, example: object = "", count: int = 1) -> None:
        self.counts[code] += count
        if example != "" and len(self.samples[code]) < 5:
            self.samples[code].append(str(example))

    def finish(self) -> None:
        self.report.problems.extend(
            AuditProblem(code, n, tuple(self.samples[code]))
            for code, n in sorted(self.counts.items())
            if n
        )
