# Local properties of common solvents and media

The bundled workbooks support local property queries without a PubChem request.
`get_medium_dataframe()` reads the 92-row medium `clean` sheet;
`get_solvent_dataframe()` reads the 227-row solvent `clean` sheet.
`query_medium()` and `query_solvent()` select existing `CASs` or integer `Cid`
identifiers, preserving request order and duplicates. Missing identifiers raise
`ValueError`; the functions do not invent values or enrich missing records.

```python
from hotpot.cheminfo.MolProps import query_medium, query_solvent

media = query_medium([176, 260, 176], id_type="Cid")
solvents = query_solvent([11, 174, 11], id_type="Cid")
solvents_by_cas = query_solvent(["107-06-2", "107-21-1"], id_type="CASs")
```

The `sub_sheet` argument is passed directly to the workbook reader. The medium
workbook also has a 96-row `fill` sheet with CAS identifiers but no `Cid` column;
`query_medium(..., sub_sheet="fill", id_type="Cid")` currently raises `KeyError`.
The solvent workbook contains `clean` and `total`, not `fill`; requesting `fill`
raises `ValueError`. These known nondefault limitations remain explicit. No
`total`/`fill` or alternative-column mapping is implied. Table presence does not
establish that every property is complete or suitable for a particular model.
