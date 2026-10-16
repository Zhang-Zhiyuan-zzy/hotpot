# Scientific-state providers

Providers are independently callable deterministic algorithms or versioned
data sources. They are not toggleable empirical ranking rules. The resolver or
a named policy decides whether to invoke an algorithm; downstream empirical
rules may consume its immutable result.

- [PROVIDER-001: supported main-group Lewis solver](PROVIDER-001-main-group-lewis-solver.md)
- [PROVIDER-002: element oxidation-state registry](PROVIDER-002-element-oxidation-registry.md)
- [PROVIDER-003: neutral isolated-atom reference data](PROVIDER-003-neutral-atom-reference.md)

Providers do not select a final execution state and do not bypass hard
validation.
