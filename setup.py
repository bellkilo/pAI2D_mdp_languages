from setuptools import setup, find_packages

setup(
    name="janiParser",
    version="0.1.0",
    author="Jiahua Li, Zeyu Tao",
    description="A tool for parsinga subset of JANI models, " \
    "solving them using Marmote library and performing benchmarking",
    url="https://github.com/bellkilo/pAI2D_mdp_languages",
    packages=find_packages(),
    install_requires=[
        "pymdptoolbox",
        "numpy",
        "scipy"
    ],
    license="MIT License"
)