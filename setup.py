from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="pyquantflow",
    version="0.2.0",
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    description="A robust stock analysis and backtesting framework.",
    packages=find_packages(include=["pyquantflow", "pyquantflow.*"]),
    install_requires=required,
)
