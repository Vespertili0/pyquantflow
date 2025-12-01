from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="stock_package",
    version="0.1.0",
    description="A robust stock analysis and backtesting framework.",
    packages=find_packages(include=["stock_package", "stock_package.*"]),
    install_requires=required,
)
