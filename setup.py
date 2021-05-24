from setuptools import find_packages, setup

requirements = [
    "pandas",
    "numpy",
    "click",
    "scikit-learn",
    "pyarrow",
    "scipy",
    "pyod",
    "geopandas",
    "folium",
    "plotly",
    "branca",
    "tqdm",
]

setup(
    name="yellowcab",
    version="0.0.1dev1",
    description="Semester Project - Programming Data Science",
    url="https://github.com/SepehrRad/PDS_TestProject_2021",
    author="Christian Bergen, Nina Annika Erlacher, Nico Liedmeyer, Sepehr Salemirad, Simon Maximilian Wolf",
    author_email="ssalemir@smail.uni-koeln.de",
    packages=find_packages(include=["yellowcab", "yellowcab.*"]),
    install_requires=requirements,
    entry_points={"console_scripts": ["yellowcab=yellowcab.cli:main"]},
    keywords="PDS",
    classifiers=[
        "Development Status :: 1 - Beta",
        "Intended Audience :: Science/Research",
        "License :: GPL-3.0 License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Education",
    ],
)
