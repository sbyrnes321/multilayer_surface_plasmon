# -*- coding: utf-8 -*-

from setuptools import setup

long_description = ("Calculates surface-plasmon-polariton modes for planar "
                    "multilayer structures. To learn more go to "
                    "http://pythonhosted.org/multilayer_surface_plasmon/")

descrip = "Calculates surface-plasmon-polariton modes for planar structures."

setup(
    name = "multilayer_surface_plasmon",
    version = "0.1.1",
    author = "Steven Byrnes",
    author_email = "steven.byrnes@gmail.com",
    description = descrip,
    license = "MIT",
    keywords = "optics, electromagnetism, surface plasmon, surface plasmon polariton",
    url = "http://pythonhosted.org/multilayer_surface_plasmon/",
    py_modules=['multilayer_surface_plasmon'],
    long_description=long_description,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        ]
)
