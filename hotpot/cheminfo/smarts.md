
# SMARTS Parser Module

This module implements the functionality to parse SMARTS (Smiles Arbitrary Target Specification) strings into `Substructure` objects. It is used for chemical substructure searching within the Hotpot project.

## 1. Design Criteria

The parser implements a **subset** of the standard SMARTS syntax, defined by the following technical constraints and logic:

*   **Flat Logical Matching**: Support is limited to "first-level" logical operations (AND `&`, `;` and OR `,`).
*   **No Recursive Environment**: Recursive SMARTS syntax `$(...)` is not supported. The parser raises a `NotImplementedError` when this syntax is encountered.
*   **Explicit Bond Inference**: Bond inference between aromatic and non-aromatic atoms follows defined rules rather than probabilistic estimation.
*   **Stream Processing**: The parsing process consists of two stages: a Tokenizer that converts the string into a stream of tokens (atoms, bonds, branches, ring markers), followed by the construction of the graph structure.

## 2. Supported Syntax

### 2.1 Basic Atoms and Bonds
- **Atom Symbols**: Standard element symbols (e.g., `C`, `N`, `P`) and aromatic lowercase symbols (e.g., `c`, `n`).
- **Wildcards**: `*` (any atom), `a` (any aromatic atom), `A` (any aliphatic atom).
- **Bond Types**:
    - `-` (single), `=` (double), `#` (triple)
    - `:` (aromatic), `~` (any bond)
    - `/`, `\` (directional single bonds)

### 2.2 Logical Operators
- **AND**: `&` or `;` (high priority). Example: `[C;H1]` matches "Carbon atom AND H count is 1".
- **OR**: `,`. Example: `[N,O]` matches "Nitrogen atom OR Oxygen atom".
- **NOT**: The `!` prefix is currently not supported.

### 2.3 Attribute Primitives
The following attributes are supported within bracket `[]` definitions:
- **Charge**: `+`, `++`, `+2`, `-`, `--`, `-2`
- **Hydrogen Count**: `H`, `H0`, `H1`, ...
- **Connectivity/Valence**: `X` (connectivity), `v` (valence electrons), `D` (explicit degree)
- **Ring Properties**: `R` (in ring), `r` (ring size), `r5` (size 5 ring)
- **Chirality**: `@`, `@@` (Parsed as a counter; geometric matching is handled by the underlying system)
- **Isotopes**: e.g., `13C`

## 3. Extended Coordination Symbols

The parser includes custom wildcard tokens extended for inorganic chemistry and material science contexts:

| Symbol | Definition | Description |
| :--- | :--- | :--- |
| **`M`** | Metal | Matches any metal atom. |
| **`!M`** | Non-Metal | Matches any non-metal atom. |
| **`Ln`** | Lanthanide | Matches Lanthanide series elements (La-Lu). |
| **`An`** | Actinide | Matches Actinide series elements (Ac-Lr). |
| **`NP<n>`** | Period | Matches elements of a specific period. Supports ranges, e.g., `NP3-5`. |
| **`NG<n>`** | Group | Matches elements of a specific group. Supports ranges, e.g., `NG1-2`. |

**Usage Examples**:
- `[M]~[O]` : Any metal atom connected to an Oxygen atom.
- `[Ln]~[N;D3]` : Any Lanthanide atom connected to a Nitrogen atom with degree 3.
- `[NP4]` : Any element in the 4th period.

## 4. API Reference

### Primary Interface

#### `substructure_from_smarts(smarts: str) -> Substructure`
The factory function for building a Substructure object.

- **Parameters**:
    - `smarts` (str): The standard or extended SMARTS pattern string.
- **Returns**:
    - `Substructure`: An object containing `QueryAtom` nodes and bond constraints.
- **Exceptions**:
    - `ValueError`: Raised for syntax errors (e.g., unmatched brackets, illegal characters).
    - `NotImplementedError`: Raised for unsupported features (e.g., recursive SMARTS `$(...)` or `!` negation).

### Internal Helpers

#### `tokenize(smarts: str) -> List[Tuple[TokenType, str]]`
Converts the raw SMARTS string into a list of tokens.
- **Token Types**: `ATOM`, `BOND`, `BRANCH_L/R`, `RING`, `BRACKET`.

#### `parse_bracket_atom(expr_text: str) -> Dict[str, object]`
Parses attribute expressions within brackets (e.g., `[C;H2;+1]`).
- **Returns**: A dictionary where keys are internal attribute names (e.g., `atomic_number`, `h_count`) and values are sets of constraints.

## 5. Bond Inference Logic

The `_infer_bond_attrs` function handles implicit bonds based on the following precedence:

1.  **Aromatic-Aromatic**: If both atoms are explicitly aromatic (e.g., `cc`), the bond defaults to an "aromatic bond" (matching single or double).
2.  **Aromatic-NonAromatic**: If one atom is aromatic and the other is explicitly non-aromatic (and no bond symbol is provided), the bond defaults to **single**.
3.  **Explicit Definition**: If a bond symbol is provided in the SMARTS string (e.g., `c-C`), that specific bond order is enforced.
