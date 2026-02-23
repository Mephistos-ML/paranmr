# SimpNMR — AI Agent Development Contract

## Usage for Chat-Based AI

When using an external AI that does not have repository access:

1. Provide the AI_CONTRACT.md file.
2. Provide the relevant directory tree.
3. Provide the specific file(s) being modified.
4. Instruct the AI that this contract overrides default behaviour.

## 0. Scope and Authority

This document defines mandatory rules for any AI system (LLM, coding agent, autocomplete tool)
generating code, tests, documentation, or patches for the SimpNMR codebase.

If any instruction here conflicts with an AI’s default behaviour, suggestions, or heuristics,
**this document takes precedence**.

Non-compliant output is considered invalid and will be rejected without review.

---

## 1. Global Non‑Negotiable Rules

1. Do not invent architecture, abstractions, or patterns.
2. Do not refactor, clean up, or reformat code unless explicitly instructed.
3. Do not change public contracts silently.
4. Do not guess scientific intent or numerical meaning.
5. Do not introduce hidden defaults, randomness, or environment‑dependent behaviour.
6. If requirements are unclear, **stop and ask for clarification**.

Public contracts include:

- CLI commands and flags
- YAML configuration schemas and key semantics
- Output file formats, column names, units, ordering, and meaning

---

## 2. Commit and Change Discipline

When generating commits or commit messages:

- Use **Conventional Commits (Angular style)**.
- Subject line must be imperative, present tense.
- Use scopes reflecting the affected subsystem (`core`, `io`, `app`, `viz`, `cli`, etc.).
- Do not manually edit version numbers or release metadata.
- Do not mark breaking changes unless explicitly instructed.

Examples:

- `feat(core): add new susceptibility model`
- `fix(io): handle malformed CSV header`
- `refactor(viz): centralise label typography`

---

## 3. Code Style and Typing

- Python only.
- Maximum line length: **88 characters**.
- Follow formatting rules defined in `pyproject.toml`.
- **Google‑style docstrings** are required for all public functions, classes, methods, and modules.
- Type hints are required for all public APIs.
- Internal helper functions may omit docstrings only if trivial and unambiguous.
- No debug print statements.
- No commented‑out code blocks.
- No dead or unused code.

All generated code must pass Ruff linting and existing tests.

---

## 4. Architectural Layers (Strict)

The codebase follows a strict layered architecture.
Allowed dependency direction only:

CLI → app → core  
    ↓  
    io, viz

Reverse or cross‑layer dependencies are forbidden.

---

### 4.1 CLI Layer (`simpnmr.cli`)

Allowed:

- Parse command‑line arguments
- Locate and load YAML configuration files
- Dispatch execution to application pipelines

Forbidden:

- Scientific logic
- Numerical routines
- File parsing beyond trivial argument handling

---

### 4.2 Application Layer (`simpnmr.app`)

Role: workflow orchestration and policy definition.

Allowed:

- Workflow wiring
- Configuration validation and routing
- Backend and method selection
- Loader coordination
- Output orchestration
- Application‑level policies (default resolution, prioritisation)

Submodules:

- `loaders`
- `pipelines`
- `policies`
- `params`

Forbidden:

- Numerical algorithms
- Scientific formulas
- Domain math or optimisation logic

---

### 4.3 Core Domain Layer (`simpnmr.core`)

This is the **only** layer where scientific and numerical logic is permitted.

Contains:

- Domain models
- Physical and chemical assumptions
- Mathematical formulas
- Optimisation routines
- Metrics and scoring
- Spin Hamiltonian logic

Constraints:

- Must be independent of CLI, IO, and visualisation
- Must be deterministic for identical inputs

---

### 4.4 IO Layer (`simpnmr.io`)

Role: translation between external formats and internal domain objects.

Allowed:

- Parsing and writing CSV, QC outputs, XYZ, plain text formats

Forbidden:

- Scientific interpretation
- Application policy
- Heuristics or implicit defaults

---

### 4.5 Visualisation Layer (`simpnmr.viz`)

Role: presentation only.

Allowed:

- Plot generation
- Centralised styling, layout, typography
- Figure export

Forbidden:

- Modifying domain objects
- Influencing numerical results
- Ad‑hoc Matplotlib code outside the unified viz infrastructure

---

## 5. Configuration and Reproducibility

- YAML schemas are part of the public API.
- No hidden defaults.
- No implicit unit conversion.
- Any randomness must be explicit, controlled, and documented.
- Same inputs and configuration must always produce the same outputs.

---

## 6. Imports and Dependency Discipline

### 6.1 Explicit imports only

- **Never use** `from x import *`.
- Always import explicit symbols.
- Prefer longer, explicit import lists over brevity.

Forbidden:

```
from simpnmr.core.sh import *
```

Required:

```
from simpnmr.core.sh import SpinHamiltonian, extract_hfc_tensor
```

---

### 6.2 Re‑export namespaces (rare exception)

`import *` is allowed **only** if all conditions are met:

1. The module exists solely to re‑export symbols.
2. The module defines an explicit `__all__`.
3. The module contains no logic.
4. The module is explicitly documented as a re‑export namespace.

Outside such modules, `import *` is prohibited.

---

### 6.3 Architectural boundary visibility

Imports must not obscure layer boundaries.

Forbidden:

- Importing `core` symbols indirectly via `app`
- Importing `viz` helpers into `core`
- Re‑exporting domain logic through convenience modules

Each layer must import directly from the layer it depends on.

---

### 6.4 Import hygiene rules

- No `import *` inside `simpnmr.app`, `simpnmr.cli`, or `simpnmr.viz`
- No circular imports
- No unused imports
- No aliasing to hide origin (`import x as y`) unless semantically justified
- Import order:
  1. standard library
  2. third‑party
  3. `simpnmr`

---

## 7. Tests and Examples

- Add tests when behaviour changes.
- Prefer unit tests for pure functions.
- Use integration tests for pipelines and workflows.
- Existing tests must not be weakened.
- Examples must remain runnable and consistent with current behaviour.

---

## 8. Extension Rules

When extending functionality:

- New QC output format → `simpnmr.io`
- New scientific or numerical model → `simpnmr.core`
- New workflow → `simpnmr.app` (+ CLI)
- Experimental or exploratory code → `simpnmr.tools` or `examples`

Avoid spreading changes across layers unless strictly required.

---

## 9. Explicit Anti‑Patterns (Never Do This)

- Broad or “helpful” refactors
- Reformatting unrelated code
- Rewriting existing APIs
- Ad‑hoc plotting in pipelines
- Convenience re‑exports that hide dependencies
- Heavyweight dependencies in `simpnmr.core` without justification
- Silent casting or implicit unit conversion
- Guessing scientific intent

---

## 10. Failure Protocol

If the AI agent cannot determine intent, required symbols, or correct placement:

**Stop. Do not guess. Ask for clarification.**

Producing speculative or assumed code is forbidden.

---

# End of Contract
