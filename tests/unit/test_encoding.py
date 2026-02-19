import pandas as pd
import pytest

from simpnmr.io.csv.csv_util import read_csv_safe, write_csv_safe


def _non_comment_lines(text: str) -> list[str]:
    """Return non-empty, non-comment lines from a CSV-like text."""
    return [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.unit
def test_write_csv_safe_emits_utf8_sig_csv(tmp_path):
    """write_csv_safe should always emit an Excel-friendly UTF-8-SIG CSV.

    This test verifies:
    - Output starts with UTF-8 BOM (because encoding defaults to utf-8-sig).
    - Output contains a generated header comment line starting with '#'.
    - CSV header and data rows are present after comment lines.
    - Non-ASCII characters survive round-trip.
    - No CR characters are present (LF-only newlines).
    """
    df = pd.DataFrame(
        {
            "col1": ["val1", "áéí"],
            "col2": ["val2", "óú"],
            "col3": ["val3", "ñ"],
        }
    )

    out_path = tmp_path / "output.csv"

    # Use defaults (utf-8-sig, newline="") to test the canonical behavior.
    write_csv_safe(df, out_path)

    raw = out_path.read_bytes()

    # UTF-8 BOM (EF BB BF) is expected for utf-8-sig.
    assert raw.startswith(b"\xef\xbb\xbf")

    # Decode with utf-8-sig to drop BOM for text assertions.
    text = raw.decode("utf-8-sig")

    # Ensure we wrote at least the generator comment.
    assert text.lstrip().startswith("# This file was generated with SimpNMR")

    # Ensure LF-only output (no CR characters).
    assert "\r" not in text

    # Extract meaningful CSV lines (skip comments/blank lines).
    lines = _non_comment_lines(text)

    # First non-comment line should be the CSV header.
    assert lines[0] == "col1,col2,col3"

    # Ensure the data rows exist and contain expected values.
    assert any("val1" in line for line in lines[1:])

    # Ensure at least one accented character survived.
    assert any(ch in text for ch in ["á", "é", "í", "ó", "ú", "ñ"])


@pytest.mark.unit
def test_read_csv_safe_roundtrip_unicode_and_comments(tmp_path):
    """read_csv_safe should correctly read CSV files produced by write_csv_safe.

    This test verifies:
    - UTF-8-SIG BOM is handled transparently.
    - Leading comment lines starting with '#' are ignored.
    - Unicode content survives a write -> read roundtrip.
    """
    df_in = pd.DataFrame(
        {
            "a": [1, 2],
            "b": ["αβγ", "ñáé"],
        }
    )

    path = tmp_path / "roundtrip.csv"

    write_csv_safe(
        df_in,
        path,
        comment=["Test comment", "Second line"],
    )

    df_out = read_csv_safe(path)

    # Column names preserved
    assert list(df_out.columns) == ["a", "b"]

    # Shape preserved
    assert df_out.shape == df_in.shape

    # Unicode content preserved
    assert df_out.loc[0, "b"] == "αβγ"
    assert df_out.loc[1, "b"] == "ñáé"
