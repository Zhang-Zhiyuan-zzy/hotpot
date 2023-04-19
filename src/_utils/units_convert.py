"""
python v3.7.9
@Project: hotpot
@File   : units_convert.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/18
@Time   : 21:18
Notes: this module is used to convert the quantity from a unit metric to other
URL: to query units conversion: https://www.convertunits.com/
the relationship among units: https://www.nist.gov/pml/owm/metric-si/si-units
"""


class UnitsConvert:
    """"""
    __path_unit_json = 'data/units.json'
    SI = {
        'mass': 'kg',
        'length': 'm',
        'temperature': 'K',
        'time': 's',
        'amount_of_substance': 'mole',
        'luminous_intensity': 'cd'
    }

    def __init__(self):
        pass
