#!/usr/bin/env python3
"""Debug CSV parsing."""

from pathlib import Path

# Find recent test output
csv_files = list(Path('/private/var/folders').rglob('*pytest*/susceptibility_tensor.csv'))
if not csv_files:
    print("No test CSV files found")
    exit(1)

csv_file = csv_files[0]
print(f"Reading: {csv_file}\n")

with open(csv_file, 'r') as f:
    lines = f.readlines()

print("First 10 lines:")
for i, line in enumerate(lines[:10]):
    print(f"{i}: {line.rstrip()}")
    
print("\n" + "="*60)
print("Now testing parser logic...")

# Test the filtering logic
filtered_lines = []
for line in lines:
    stripped = line.strip()
    if stripped and not stripped.startswith('#'):
        filtered_lines.append(line)

print(f"\nFiltered to {len(filtered_lines)} non-comment lines")
print("First 3 filtered lines:")
for i, line in enumerate(filtered_lines[:3]):
    print(f"{i}: {line.rstrip()}")

# Test CSV parsing
import csv
import io

reader = csv.DictReader(io.StringIO(''.join(filtered_lines)))
print(f"\nHeaders: {reader.fieldnames}")

print("\nFirst 3 rows as parsed:")
for i, row in enumerate(reader):
    if i < 3:
        print(f"Row {i}:")
        print(f"  Temperature (K) = '{row['Temperature (K)']}'")
        print(f"  chi_iso (Å^3) = '{row['chi_iso (Å^3)']}'")
