from setuptools import setup, find_packages

setup(
    name="spectra_analyser",
    version="0.1.0",
    description="",
    author="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=["cli", "config"],
    install_requires=[
        "numpy",
        "pyteomics",
        "matchms",
    ],
    entry_points={
        "console_scripts": [
            "spectra_analyser = cli:main",
        ],
    },
)
