# Querying properties of common Solvents and Media in liquid-liquid extraction

This toy module defines two interfaces (`query_media`) and (`query_solvents`) to access the 90 media and 227 solvents

### Utils with example

```pycon
import pandas as pd
from hotpot.cheminfo.MolProps import query_medium, query_solvent

list_cas = ['50-00-0', '64-17-5']
list_cid = [11, 11, 174]

df: pd.DataFrame = query_medium(list_cid, id_type='Cid')  # Query media by Cids
print(df)

df: pd.DataFrame = query_solent(list_cid, id_type='Cid')  # Query solvents by Cids
print(df)

df: pd.DataFrame = query_solent(list_cas, id_type='CASs')  # Query solvents by CASs
print(df)

df: pd.DataFrame = query_solent(list_cas, id_type='CASs')  # Query solvents by CASs
print(df)
```
