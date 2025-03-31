from typing import Literal

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

    def __call__(self, text):
        print(self.fmt + text + self.reset)

    @classmethod
    def from_hex(cls, hex_color: str, style: PrintStyle = 'regular') -> 'FmtPrint':
        return cls(hex_to_ansi(hex_color, style))


light_green = FmtPrint('\033[92m')
dark_green = FmtPrint('\033[32m')
bold_dark_green = FmtPrint('\033[1m')

bold_orange = FmtPrint('\033[38;5;208m')
bold_magenta = FmtPrint('\033[1;35m')

__all__ = [k for k, v in locals().items() if isinstance(v, FmtPrint)]
