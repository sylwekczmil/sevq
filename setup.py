#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy>=1.18.5', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Sylwester Czmil",
    author_email='sylwekczmil@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="SEVQ: Simplified Evolving Vector Quantization",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='sevq',
    name='sevq',
    packages=find_packages(include=['sevq', 'sevq.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/sylwekczmil/sevq',
    version='1.0.0',
    zip_safe=False,
)
