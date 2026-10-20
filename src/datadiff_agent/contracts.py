"""Dataset contracts: the business context the agent reasons with.

A raw diff says *what* changed. A contract says *whether it matters*:
which columns are critical, what each column means, which reports depend
on the dataset, and which thresholds apply. The agent reads the contract
through a tool, and the guardrails in ``policy.py`` use it too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from datadiff_engine import DiffConfig


@dataclass(frozen=True)
class ColumnContract:
    name: str
    description: str = ""
    critical: bool = False
    # Columns such as IDs where numeric drift is meaningless.
    identifier: bool = False


@dataclass(frozen=True)
class DatasetContract:
    name: str
    description: str = ""
    owner: str = ""
    downstream: tuple[str, ...] = ()
    columns: dict[str, ColumnContract] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def critical_columns(self) -> set[str]:
        return {c.name for c in self.columns.values() if c.critical}

    def diff_config(self) -> DiffConfig:
        """Build the datadiff-engine config for this dataset."""
        ignored = frozenset(
            c.name for c in self.columns.values() if c.identifier
        )
        return DiffConfig(ignored_columns=ignored, **self.thresholds)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "downstream": list(self.downstream),
            "critical_columns": sorted(self.critical_columns),
            "columns": {
                c.name: {
                    "description": c.description,
                    "critical": c.critical,
                    "identifier": c.identifier,
                }
                for c in self.columns.values()
            },
            "thresholds": self.thresholds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetContract":
        columns = {
            name: ColumnContract(name=name, **spec)
            for name, spec in data.get("columns", {}).items()
        }
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            owner=data.get("owner", ""),
            downstream=tuple(data.get("downstream", [])),
            columns=columns,
            thresholds=data.get("thresholds", {}),
        )


def load_contract(path_or_name: str | Path) -> DatasetContract:
    """Load a contract from a JSON file, or a bundled one by name."""
    path = Path(path_or_name)
    if path.suffix == ".json" and path.exists():
        return DatasetContract.from_dict(json.loads(path.read_text()))

    bundled = resources.files("datadiff_agent").joinpath(
        "contract_specs", f"{path_or_name}.json"
    )
    if bundled.is_file():
        return DatasetContract.from_dict(json.loads(bundled.read_text()))

    raise FileNotFoundError(
        f"No contract found for '{path_or_name}'. Pass a path to a "
        "contract JSON file or the name of a bundled contract."
    )


def default_contract(name: str) -> DatasetContract:
    """A minimal contract used when a dataset has none."""
    return DatasetContract(
        name=name,
        description="No contract provided; all columns treated as non-critical.",
    )
