class FmtPrint:
    def __init__(self, fmt: str):
        self.fmt = fmt
        self.reset = '\033[0m'

    def __call__(self, text):
        print(self.fmt + text + self.reset)


light_green = FmtPrint('\033[92m')
dark_green = FmtPrint('\033[32m')




__all__ = [k for k, v in locals().items() if isinstance(v, FmtPrint)]
