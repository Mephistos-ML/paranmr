# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Define the YAML schema for the ``plot_chit`` workflow."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from simpnmr_x.cfg.config import Config


@dataclass(frozen=True)
class ChiTSourceConfig:
    """Configuration for one ORCA ``chi*T`` source file."""

    file: str
    format: str
    section: str


@dataclass(frozen=True)
class ChiTTipConfig:
    """Configuration for analytical TIP removal."""

    mode: str
    reference_temperature: str | float


@dataclass(frozen=True)
class ChiTTemperatureConfig:
    """Temperature limits for the χT plot, in kelvin."""

    minimum: float
    maximum: float


class PlotChiTConfig(Config):
    """Validate and expose configuration for ``plot_chit``."""

    REQ_KEYWORDS = {"plot_chit": ["output"]}
    KEYWORDS = {"plot_chit": ["xrd", "opt", "tip", "temperature", "output"]}
    KEYWORD_PARTNERS: dict[str, list[str]] = {}

    def __init__(self, **kwargs) -> None:
        self._xrd: ChiTSourceConfig | None = None
        self._opt: ChiTSourceConfig | None = None
        self._tip: ChiTTipConfig | None = None
        self._temperature: ChiTTemperatureConfig | None = None
        self._output_file = ""

        for keyword, value in kwargs.items():
            short_keyword = keyword.removeprefix("plot_chit_")
            if short_keyword == "output":
                short_keyword = "output_file"
            setattr(self, short_keyword, value)

        if self._xrd is None and self._opt is None:
            raise ValueError("plot_chit requires at least one of xrd or opt")
        if self._tip is not None and self._opt is None:
            raise ValueError("plot_chit:tip requires an opt source")

    @property
    def xrd(self) -> ChiTSourceConfig | None:
        """Return the optional XRD susceptibility source."""

        return self._xrd

    @xrd.setter
    def xrd(self, value: dict) -> None:
        self._xrd = self._parse_source(value, "xrd")

    @property
    def opt(self) -> ChiTSourceConfig | None:
        """Return the optional OPT susceptibility source."""

        return self._opt

    @opt.setter
    def opt(self, value: dict) -> None:
        self._opt = self._parse_source(value, "opt")

    @property
    def tip(self) -> ChiTTipConfig | None:
        """Return the optional TIP-removal configuration."""

        return self._tip

    @tip.setter
    def tip(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError("plot_chit:tip must be a mapping")

        mode = value.get("mode")
        if mode != "analytic":
            raise ValueError("plot_chit:tip:mode must be 'analytic'")

        reference_temperature = value.get("reference_temperature")
        if reference_temperature is None:
            raise KeyError("plot_chit:tip requires reference_temperature")
        if isinstance(reference_temperature, str):
            if reference_temperature != "max":
                raise ValueError(
                    "plot_chit:tip:reference_temperature must be 'max' or a number"
                )
        else:
            reference_temperature = float(reference_temperature)
            if reference_temperature <= 0.0:
                raise ValueError("plot_chit:tip:reference_temperature must be positive")

        self._tip = ChiTTipConfig(
            mode=mode,
            reference_temperature=reference_temperature,
        )

    @property
    def temperature(self) -> ChiTTemperatureConfig | None:
        """Return the optional temperature limits for the plot."""

        return self._temperature

    @temperature.setter
    def temperature(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError("plot_chit:temperature must be a mapping")

        try:
            minimum = float(value["min"])
            maximum = float(value["max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "plot_chit:temperature requires numeric min and max"
            ) from exc

        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("plot_chit:temperature limits must be finite")
        if minimum <= 0.0:
            raise ValueError("plot_chit:temperature:min must be positive")
        if maximum <= minimum:
            raise ValueError("plot_chit:temperature:max must be greater than min")

        self._temperature = ChiTTemperatureConfig(
            minimum=minimum,
            maximum=maximum,
        )

    @property
    def output_file(self) -> str:
        """Return the output PDF path resolved from the working directory."""

        return self._output_file

    @output_file.setter
    def output_file(self, value: dict) -> None:
        if not isinstance(value, dict) or not isinstance(value.get("file"), str):
            raise TypeError("plot_chit:output must contain a string file")
        if not value["file"].strip():
            raise ValueError("plot_chit:output:file must not be empty")
        self._output_file = os.path.abspath(value["file"])

    @staticmethod
    def _parse_source(value: dict, name: str) -> ChiTSourceConfig:
        if not isinstance(value, dict):
            raise TypeError(f"plot_chit:{name} must be a mapping")

        file_name = value.get("file")
        file_format = value.get("format")
        section = value.get("section")
        if not all(isinstance(item, str) for item in (file_name, file_format, section)):
            raise TypeError(
                f"plot_chit:{name} requires string file, format, and section"
            )
        if file_format.lower() != "orca":
            raise ValueError(f"plot_chit:{name}:format must be 'orca'")
        if section.lower() not in {"casscf", "nevpt2"}:
            raise ValueError(f"plot_chit:{name}:section must be 'casscf' or 'nevpt2'")
        if not file_name.strip():
            raise ValueError(f"plot_chit:{name}:file must not be empty")

        return ChiTSourceConfig(
            file=os.path.abspath(file_name),
            format=file_format.lower(),
            section=section.lower(),
        )
