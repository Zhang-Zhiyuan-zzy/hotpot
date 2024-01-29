"""
python v3.9.0
@Project: hotpot
@File   : _io
@Auther : Zhiyuan Zhang
@Data   : 2024/1/25
@Time   : 9:57
"""
from typing import Union
from pathlib import Path


def parse_gjf(path_gjf: Union[str, Path]):
    """"""
    def _parse_link0():
        nonlocal state
        if line.startswith('%'):
            parsed_info['link0'].append(line.strip())
        else:
            state += 1  # into route

    def _parse_route():
        nonlocal state
        if line.startswith('#'):
            parsed_info['route'].append(line.strip())
        else:
            state += 1  # into title

    def _parse_title():
        nonlocal state
        if line.strip():
            parsed_info['title'] = line.strip()
            state += 1  # into charge_spin

    def _parse_charge_spin():
        nonlocal state
        if line.strip():
            parsed_info['charge'], parsed_info['spin'] = map(int, line.strip().split())
            state += 1  # into coordination

    def _parse_coordination():
        nonlocal state
        if line.strip():
            parsed_info['coordinates'].append(line.strip())
        else:
            state += 1  # into addition

    def _parse_addition():
        parsed_info['addition'].append(line)

    with open(path_gjf) as file:
        lines = file.readlines()

    parsed_info = {
        'link0': [],
        'route': [],
        'title': '',
        'charge': None,
        'spin': None,
        'coordinates': [],
        'addition': []
    }

    state = 0
    for line in lines:

        if state == 0:
            _parse_link0()
        if state == 1:
            _parse_route()
        if state == 2:
            _parse_title()
        elif state == 3:
            _parse_charge_spin()
        elif state == 4:
            _parse_coordination()
        elif state == 5:
            _parse_addition()

    return parsed_info


def reorganize_gjf(parsed_gjf: dict):
    script = ""
    for line in parsed_gjf['link0']:
        script += line + '\n'
    for line in parsed_gjf['route']:
        script += line + '\n'
    script += f'\n{parsed_gjf["title"]}\n\n'
    script += f'{parsed_gjf["charge"]} {parsed_gjf["spin"]}\n'
    for line in parsed_gjf['coordinates']:
        script += line + '\n'
    script += '\n'
    for line in parsed_gjf['addition']:
        script += line

    return script


if __name__ == '__main__':
    data = parse_gjf('/mnt/c/Users/zhang/OneDrive/Papers/Gibbs with logK/results/g16/gjf/pairs/81_81_C20H28N2O6P2Am.gjf')
    s = reorganize_gjf(data)
