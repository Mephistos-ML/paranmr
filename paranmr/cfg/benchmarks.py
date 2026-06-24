# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define YAML-backed configuration schemas for benchmark workflows.

The hyperfine benchmark configuration is intentionally separate from the main
workflow configuration module. It accepts a project block and an explicit list
of hyperfine input blocks:

```

```
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import yaml_include


@dataclass(frozen=True)
class HyperfineBenchmarkBlock:
    """Input source for an A_fc benchmark run.

    Args:
        functional: Functional name associated with the benchmark source.
        file: Path to the hyperfine source file.
        method: Hyperfine source type. Supported values are ``"dft"`` and
            ``"csv"``. Defaults to ``"dft"``.

    Raises:
        ValueError: If `functional`, `file`, or `method` is invalid.
    """

    functional: str
    file: str
    method: str = "dft"

    def __post_init__(self) -> None:
        if not isinstance(self.functional, str) or not self.functional:
            raise ValueError("hyperfine:functional must be a non-empty string")
        if self.method not in {"dft", "csv"}:
            raise ValueError(
                "Unknown hyperfine:method "
                f"{self.method!r}. Expected one of {'dft', 'csv'}."
            )
        if not isinstance(self.file, str) or not self.file:
            raise ValueError("hyperfine:file must be a non-empty string")

        object.__setattr__(self, "file", str(Path(self.file).expanduser().resolve()))


class AfcBenchmarkConfig:
    """Configuration for the ``paranmr benchmark a_fc`` workflow.

    Args:
        project_name: Output project directory name.
        signal_labels_file: Path to the signal-label mapping file.
        nuclei_include: Nuclei selected for the benchmark.
        max_label_tolerance: Relative tolerance for replacing one-off max labels
            with the majority max label in functional max plots.
        hyperfine: Hyperfine input blocks to parse and benchmark.

    Raises:
        ValueError: If required values are missing or malformed.
    """

    KEYWORDS = {
        "project": ["name"],
        "signal_labels": ["file"],
        "nuclei": ["include"],
        "benchmark": ["max_label_tolerance"],
        "hyperfine": ["functional", "method", "file"],
    }
    REQ_KEYWORDS = {
        "project": ["name"],
        "signal_labels": ["file"],
        "nuclei": ["include"],
        "hyperfine": ["functional", "file"],
    }

    def __init__(
        self,
        project_name: str,
        signal_labels_file: str,
        nuclei_include: list[str] | str,
        max_label_tolerance: float,
        hyperfine: list[HyperfineBenchmarkBlock],
    ) -> None:
        self.project_name = project_name
        self.signal_labels_file = signal_labels_file
        self.nuclei_include = nuclei_include
        self.max_label_tolerance = max_label_tolerance
        self.hyperfine = hyperfine

    @classmethod
    def from_file(cls, file_name: str) -> "AfcBenchmarkConfig":
        """Create an A_fc benchmark configuration from a YAML file.

        Args:
            file_name: Path to the YAML configuration file.

        Returns:
            Parsed A_fc benchmark configuration.

        Raises:
            KeyError: If required blocks or keys are missing, or if unsupported
                keys are present.
            ValueError: If a block has an invalid type or value.
            yaml.YAMLError: If the YAML file has invalid syntax or structure.
        """

        yaml.add_constructor("!inc", yaml_include.Constructor(base_dir="."))

        try:
            with open(file_name, "r", encoding="utf-8") as input_file:
                parsed = yaml.full_load(input_file)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(
                f"Invalid YAML structure in input file '{file_name}'."
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError("Benchmark configuration must be a YAML mapping")

        parsed = cls._apply_master_block(parsed)
        cls._validate_top_level_keys(parsed)

        project_name = cls._parse_project_name(parsed["project"])
        signal_labels_file = cls._parse_signal_labels_file(parsed["signal_labels"])
        nuclei_include = cls._parse_nuclei_include(parsed["nuclei"])
        max_label_tolerance = cls._parse_max_label_tolerance(
            parsed.get("benchmark", {})
        )
        hyperfine_blocks = cls._parse_hyperfine_blocks(parsed["hyperfine"])

        return cls(
            project_name=project_name,
            signal_labels_file=signal_labels_file,
            nuclei_include=nuclei_include,
            max_label_tolerance=max_label_tolerance,
            hyperfine=hyperfine_blocks,
        )

    @staticmethod
    def _apply_master_block(parsed: dict[str, Any]) -> dict[str, Any]:
        """Apply legacy-style ``master`` overrides used by existing configs."""
        merged = dict(parsed)
        master = merged.pop("master", None)
        if master is None:
            return merged
        if not isinstance(master, dict):
            raise ValueError("master block must be a mapping")
        for key, value in master.items():
            merged[key] = value
        return merged

    @classmethod
    def _validate_top_level_keys(cls, parsed: dict[str, Any]) -> None:
        """Validate supported and required top-level YAML keys."""
        unsupported = [key for key in parsed if key not in cls.KEYWORDS]
        if unsupported:
            raise KeyError(f"Unsupported benchmark config keyword(s): {unsupported}")

        missing = [key for key in cls.REQ_KEYWORDS if key not in parsed]
        if missing:
            raise KeyError(f"Missing benchmark config keyword(s): {missing}")

    @classmethod
    def _parse_project_name(cls, project_block: Any) -> str:
        """Parse and validate the ``project`` YAML block."""
        if not isinstance(project_block, dict):
            raise ValueError("project block must be a mapping")

        cls._validate_block_keys("project", project_block)

        if "name" not in project_block:
            raise KeyError("Error: missing keyword project:name")

        project_name = project_block["name"]
        if not isinstance(project_name, str) or not project_name:
            raise ValueError("project:name must be a non-empty string")

        return project_name

    @classmethod
    def _parse_signal_labels_file(cls, signal_labels_block: Any) -> str:
        """Parse and validate the ``signal_labels`` YAML block."""
        if not isinstance(signal_labels_block, dict):
            raise ValueError("signal_labels block must be a mapping")

        cls._validate_block_keys("signal_labels", signal_labels_block)

        if "file" not in signal_labels_block:
            raise KeyError("Error: missing keyword signal_labels:file")

        signal_labels_file = signal_labels_block["file"]
        if not isinstance(signal_labels_file, str) or not signal_labels_file:
            raise ValueError("signal_labels:file must be a non-empty string")

        return str(Path(signal_labels_file).expanduser().resolve())

    @classmethod
    def _parse_nuclei_include(cls, nuclei_block: Any) -> list[str] | str:
        """Parse and validate the ``nuclei`` YAML block."""
        if not isinstance(nuclei_block, dict):
            raise ValueError("nuclei block must be a mapping")

        cls._validate_block_keys("nuclei", nuclei_block)

        if "include" not in nuclei_block:
            raise KeyError("Error: missing keyword nuclei:include")

        nuclei_include = nuclei_block["include"]
        if isinstance(nuclei_include, str):
            if not nuclei_include:
                raise ValueError("nuclei:include must be a non-empty string or list")
            return nuclei_include

        if isinstance(nuclei_include, list):
            if not nuclei_include:
                raise ValueError("nuclei:include must contain at least one nucleus")
            if not all(
                isinstance(nucleus, str) and nucleus for nucleus in nuclei_include
            ):
                raise ValueError("nuclei:include must contain only non-empty strings")
            return nuclei_include

        raise ValueError("nuclei:include must be a non-empty string or list")

    @classmethod
    def _parse_max_label_tolerance(cls, benchmark_block: Any) -> float:
        """Parse and validate ``benchmark:max_label_tolerance``."""
        if benchmark_block is None:
            return 0.0
        if not isinstance(benchmark_block, dict):
            raise ValueError("benchmark block must be a mapping")

        cls._validate_block_keys("benchmark", benchmark_block)

        value = benchmark_block.get("max_label_tolerance", 0.0)
        try:
            tolerance = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "benchmark:max_label_tolerance must be numeric"
            ) from exc

        if tolerance < 0:
            raise ValueError("benchmark:max_label_tolerance must be non-negative")

        return tolerance

    @classmethod
    def _parse_hyperfine_blocks(
        cls,
        hyperfine_blocks: Any,
    ) -> list[HyperfineBenchmarkBlock]:
        """Parse and validate the repeated ``hyperfine`` YAML blocks."""
        if not isinstance(hyperfine_blocks, list):
            raise ValueError("hyperfine block must be a list of mappings")
        if not hyperfine_blocks:
            raise ValueError("hyperfine block must contain at least one input")

        parsed_blocks: list[HyperfineBenchmarkBlock] = []
        for index, block in enumerate(hyperfine_blocks):
            if not isinstance(block, dict):
                raise ValueError(f"hyperfine entry at index {index} must be a mapping")
            cls._validate_block_keys("hyperfine", block)

            for subkeyword in cls.REQ_KEYWORDS["hyperfine"]:
                if subkeyword not in block:
                    raise KeyError(
                        f"Error: missing keyword hyperfine[{index}]:{subkeyword}"
                    )

            parsed_blocks.append(
                HyperfineBenchmarkBlock(
                    functional=block["functional"],
                    file=block["file"],
                    method=block.get("method", "dft"),
                )
            )

        return parsed_blocks

    @classmethod
    def _validate_block_keys(cls, block_name: str, block: dict[str, Any]) -> None:
        """Validate supported subkeys for a YAML block."""
        unsupported = [key for key in block if key not in cls.KEYWORDS[block_name]]
        if unsupported:
            raise KeyError(
                f"Unsupported benchmark config keyword(s) in {block_name}: "
                f"{unsupported}"
            )

    @property
    def project_name(self) -> str:
        """Benchmark output project directory name."""
        return self._project_name

    @project_name.setter
    def project_name(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("project_name must be a non-empty string")
        self._project_name = value

    @property
    def signal_labels_file(self) -> str:
        """Path to the signal-label mapping file."""
        return self._signal_labels_file

    @signal_labels_file.setter
    def signal_labels_file(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("signal_labels_file must be a non-empty string")
        self._signal_labels_file = str(Path(value).expanduser().resolve())

    @property
    def nuclei_include(self) -> list[str] | str:
        """Nuclei selected for the benchmark run."""
        return self._nuclei_include

    @nuclei_include.setter
    def nuclei_include(self, value: list[str] | str) -> None:
        if isinstance(value, str):
            if not value:
                raise ValueError("nuclei_include must be a non-empty string or list")
            self._nuclei_include = value
            return

        if isinstance(value, list):
            if not value:
                raise ValueError("nuclei_include must contain at least one nucleus")
            if not all(isinstance(nucleus, str) and nucleus for nucleus in value):
                raise ValueError("nuclei_include must contain only non-empty strings")
            self._nuclei_include = value
            return

        raise ValueError("nuclei_include must be a non-empty string or list")

    @property
    def max_label_tolerance(self) -> float:
        """Relative tolerance for majority-label max replacement."""
        return self._max_label_tolerance

    @max_label_tolerance.setter
    def max_label_tolerance(self, value: float) -> None:
        try:
            tolerance = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_label_tolerance must be numeric") from exc
        if tolerance < 0:
            raise ValueError("max_label_tolerance must be non-negative")
        self._max_label_tolerance = tolerance

    @property
    def hyperfine(self) -> list[HyperfineBenchmarkBlock]:
        """Hyperfine input blocks for the benchmark run."""
        return self._hyperfine

    @hyperfine.setter
    def hyperfine(self, value: list[HyperfineBenchmarkBlock]) -> None:
        if not isinstance(value, list) or not value:
            raise ValueError("hyperfine must be a non-empty list")
        if not all(isinstance(item, HyperfineBenchmarkBlock) for item in value):
            raise ValueError(
                "hyperfine must contain only HyperfineBenchmarkBlock entries"
            )
        self._hyperfine = value

    @property
    def hyperfine_methods(self) -> list[str]:
        """Hyperfine methods in config order."""
        return [block.method for block in self.hyperfine]

    @property
    def hyperfine_files(self) -> list[str]:
        """Hyperfine files in config order."""
        return [block.file for block in self.hyperfine]


class AsdBenchmarkConfig(AfcBenchmarkConfig):
    """Configuration for the ``paranmr benchmark a_sd`` workflow."""
