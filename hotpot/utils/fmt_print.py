import io
from typing import Literal, Union

from rich.console import Console
from rich.table import Table

# Determine style codes
_style_codes = {
    'regular': '',
    'bold': '1;',
    'italic': '3;',
    'underline': '4;',
    'blink': '5;',
    'reverse': '7;',
    'hidden': '8;'
}
PrintStyle = Literal['regular', 'bold', 'italic', 'underline', 'blink', 'reverse', 'hidden']
def hex_to_ansi(hex_color: str, style: PrintStyle = 'regular') -> str:
    # Remove the '#' if present
    hex_color = hex_color.lstrip('#')

    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # ANSI escape code for 256 colors
    ansi_color = 16 + (36 * (r // 51)) + (6 * (g // 51)) + (b // 51)

    # Get the style code, default to regular if not found
    style_code = _style_codes.get(style, '')

    return f"\033[{style_code}38;5;{ansi_color}m"


class FmtPrint:
    def __init__(self, fmt: str):
        self.fmt = fmt
        self.reset = '\033[0m'

    def __call__(self, contents):
        print(self.fmt + str(contents) + self.reset)

    @classmethod
    def from_hex(cls, hex_color: str, style: PrintStyle = 'regular') -> 'FmtPrint':
        return cls(hex_to_ansi(hex_color, style))


light_green = FmtPrint('\033[92m')
dark_green = FmtPrint('\033[32m')
bold_dark_green = FmtPrint('\033[1m')

bold_orange = FmtPrint('\033[38;5;208m')
bold_magenta = FmtPrint('\033[1;35m')

def dict_to_table(
        metrics_dict: dict[str, Union[float, dict[str, float]]],
        table_kw: dict = None,
        title: str = None,
):
    def _new_row():
        nonlocal row
        if len(row) == 4 * t_cols:
            rows.append(row)
            row = []

    if table_kw is None:
        table_kw = {}

    assert len(metrics_dict) > 0
    t_cols = min(4, len(metrics_dict))
    t_rest = len(metrics_dict) % t_cols

    table = Table(title=title, **table_kw)
    for _ in range(t_cols):
        table.add_column('ID', no_wrap=True, min_width=3)
        table.add_column('Task', no_wrap=True, min_width=5)
        table.add_column('Metric', no_wrap=True, min_width=10)
        table.add_column('Value', no_wrap=True, min_width=12)

    rows = []
    row = []
    for i, (tsk, dict_value) in enumerate(metrics_dict.items(), 1):
        if isinstance(dict_value, dict):
            for mtrc_name, mtrc_value in dict_value.items():
                _new_row()
                row.extend(map(str, (i, tsk, mtrc_name, f'{mtrc_value:.3g}')))
        else:
            _new_row()
            row.extend(map(str, (i, tsk, tsk, f'{dict_value:.3g}')))

    for _ in range(t_rest):
        row.extend([''] * 4)
    rows.append(row)

    for row in rows:
        table.add_row(*row)

    return table

def export_table(table: Table, save_path: str):
    # Capture table output to a string
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, color_system=None)
    console.print(table)

    # Write string to a text file
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())

def rich_print(contents, **kwargs):
    console = Console(**kwargs)
    console.print(contents)


__all__ = [
    'dict_to_table',
] + [k for k, v in locals().items() if isinstance(v, FmtPrint)]
