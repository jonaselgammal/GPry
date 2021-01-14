#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='gpry',
    version='0.0.1',
    description='Python GPry Package',
    long_description="""The documentation can be found
    at ...""",
    long_description_content_type='text/markdown',
    author='Jonas El Gammal',
    author_email='jonas.el.gammal@rwth-aachen.de',
    license='MIT',
    keywords='Gaussian Processes',
    url='...',
    project_urls={
        'Documentation':
        '...',
        'Source': '...',
        },
    packages=['gpry'],
    package_data={'gpry': ['data/*']},
    classifiers=[
        'Development Status :: 4 - alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        ],
    install_requires=['numpy', 'scikit-learn']
)
