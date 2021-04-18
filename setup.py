from setuptools import setup

setup(
    name='yellowcab',
    version='0.0.1dev1',
    description="Semester Project - Programming Data Science",
    author="Student",
    author_email="student@uni-koeln.de",
    packages=["yellowcab"],
    install_requires=[
        'pandas',
        'click',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': ['yellowcab=yellowcab.cli:main']
    }
)
