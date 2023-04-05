import json


feature_names = ['melt', 'density']
with open('/home/dy/hotpot/data/periodic_table.json') as f:
    element_data = json.load(f)

features = {}
for name in feature_names:
    if name in element_data:
        features[name] = element_data[name]