"""
python v3.9.0
@Project: hotpot
@File   : setup
@Auther : Zhiyuan Zhang
@Data   : 2023/7/21
@Time   : 5:25
"""
import setuptools


setuptools.setup(
    name="hotpot-zzy",
    version="0.5.0.0",
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
