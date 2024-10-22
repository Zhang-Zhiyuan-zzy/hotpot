"""
python v3.9.0
@Project: hotpot
@File   : setup
@Auther : Zhiyuan Zhang
@Data   : 2023/7/21
@Time   : 5:25
"""
import re
import shutil
from os import path
import setuptools


__src_dir__ = path.dirname(__file__)

# Read the version from __version__.py
def get_version():
    version = {}

    try:
        with open(path.join(__src_dir__, 'hotpot', "__version__.py"), 'r') as f:
            exec(f.read(), version)

    except FileNotFoundError:
        return None

    return version['__version__']

__version__ = get_version()
print(__version__)


def update_pyproject_version():
    with open(path.join(__src_dir__, "pyproject.toml"), 'r') as f:
        content = f.read()

    # Use regex to find the version line and replace it
    new_content = re.sub(r'version = "\d+\.\d+\.\d+\.\d+"', f'version = "{__version__}"', content)

    print(new_content)
    # with open(pyproject_file, 'w') as f:
    #     f.write(new_content)


update_pyproject_version()

setuptools.setup(
    name="hotpot-zzy",
    version=__version__,
    description="A python package designed to communicate among various chemical and materials calculational tools",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    author="Zhiyuan Zhang",
    author_email="ZhiyuanZhang_scu@163.com",
    url="https://github.com/Zhang-Zhiyuan-zzy/hotpot",
    project_urls={
        "Homepage": "https://github.com/Zhang-Zhiyuan-zzy/hotpot"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "thermo",
        "cclib",
        "tqdm",
        "rdkit",
        "psutil",
        "dpdata"
    ],
    include_package_data=True,
)
