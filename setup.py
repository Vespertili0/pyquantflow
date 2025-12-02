from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="pyquantflow",
    version="0.2.0",
    description="A robust stock analysis and backtesting framework.",
    packages=find_packages(include=["pyquantflow", "pyquantflow.*"]),
    install_requires=required,
)
