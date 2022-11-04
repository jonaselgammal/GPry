#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from os import path

def get_long_description():
    """Get the long description from the README file."""
    with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'),
              encoding='utf-8') as f:
        lines = f.readlines()
        return "".join(lines[:])


setup(
    name='gpry',
    version='1.1.0',
    description='A package for fast bayesian inference of expensive Likelihoods',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Jonas El Gammal, Jesus Torrado, Nils Schoeneberg and Christian Fidler',
    author_email='jonas.el.gammal@rwth-aachen.de',
    license='LGPL',
    keywords=['inference', 'gaussianprocesses', 'sampling', 'cosmology'],
    url='https://github.com/jonaselgammal/GPry',
    project_urls={
        'Documentation':
        'https://gpry.readthedocs.io',
        'Source': 'https://github.com/jonaselgammal/GPry',
        },
    packages=['gpry'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    install_requires=['cobaya>=3.2.2', 'scikit-learn', 'dill', 'mpi4py']
)
