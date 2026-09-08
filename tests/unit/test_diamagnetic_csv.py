"""Tests for atom- and signal-resolved diamagnetic CSV loading."""

from __future__ import annotations

import pytest

from paranmr.app.loaders.dia_load import load_diamagnetic_shifts


def test_load_diamagnetic_csv_by_atom_label(tmp_path):
    file_name = tmp_path / "dia.csv"
    file_name.write_text("atom_label,shift\nH1,1.25\nH2,2.5\n", encoding="utf-8")

    shifts, key_kind, reference = load_diamagnetic_shifts(file_name=str(file_name))

    assert shifts == {"H1": 1.25, "H2": 2.5}
    assert key_kind == "atom_label"
    assert reference is None


def test_load_diamagnetic_csv_rejects_ambiguous_key_columns(tmp_path):
    file_name = tmp_path / "dia.csv"
    file_name.write_text(
        "atom_label,signal_label,shift\nH1,MeA,1.25\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exactly one key column"):
        load_diamagnetic_shifts(file_name=str(file_name))
