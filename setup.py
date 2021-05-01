from setuptools import find_packages, setup

setup(
    name="yellowcab",
    version="0.0.1dev1",
    description="Semester Project - Programming Data Science",
    author="Student",
    author_email="student@uni-koeln.de",
    packages=find_packages(include=["yellowcab", "yellowcab.*"]),
    install_requires=["pandas", "numpy", "click", "scikit-learn", "pyarrow"],
    entry_points={"console_scripts": ["yellowcab=yellowcab.cli:main"]},
)
