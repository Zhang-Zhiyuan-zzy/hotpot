# Stage12A explicit facade and lazy toolkit boundary

Root accepts this concrete Decision21 route on 2026-11-04 at the entry source
commit before authoring. It implements main-plan §§7.1/9.3/9.4 without a new
chemical API, alternate runtime or scientific algorithm.

1. `hotpot.__all__` is the current explicit `cheminfo.__all__` sequence plus
   the retained `version` callable. Root imports those names explicitly from
   their existing facade; each chemical object's identity remains unchanged.
   `version()` reads the one existing version constant. Retire wildcard export,
   RDKit preload, sys.path mutation, global tempdir and package_root; imported
   os/sys/path helpers are no longer public conveniences.
2. Importing Hotpot and obtaining those class/function declarations must not
   import RDKit/Open Babel or inference runtimes. Existing annotation/default
   objects and canonical native loaders remain with their current owners.
3. Known Open Babel/PyPI library-order handling moves to one private external
   runtime leaf under adapters/openbabel. On the first Hotpot Open Babel use,
   load RDKit Chem before Open Babel/pybel. Reuse normal Python import caching;
   add no registry, fallback, exception suppression or eagerly loaded toolkit.
   Migrate the actual Open Babel adapter and standalone-wrapper import sites
   that need this order together, including the existing `works.convert` pybel
   import; this is an import-routing correction, not command relocation. Direct
   unmanaged third-party-first imports
   remain an external diagnostic, not a Hotpot acceptance requirement.
4. The CCDC solvent module uses importlib.resources/as_file over its existing
   packaged solvent files; importing the module loads neither CCDC nor solvent
   molecules. First access to `solvents` or `solvents_smiles` calculates the
   existing OrderedDict/set once. Missing CCDC exposes its original import
   failure; no empty-result fallback. This changes loading time, not the data
   or its ownership. Other Decision13 data moves remain pending. The example's
   source-repository output path uses its own file location rather than
   hotpot.package_root; it is not an installed-package resource.
5. Source/installed fresh processes must demonstrate no toolkit eager import,
   no sys.path addition, exact approved exports/class identity/version and both
   actual adapter orders/native coexistence. Record actual graph/charge values
   and native origins. Mocked CCDC resource tests prove this adapter's loading
   contract only; unavailable real CCDC remains unverified. Retain all failures.

No new DTO, chemical fact, model tensor, native source or ABI change is accepted.
The entry is a bounded Stage12A route, not closure of P20-D026, other Stage12 work,
whole Stage11, versions/release, Plan12 or global scientific/performance gates.
