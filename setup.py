import os
from distutils.core import setup

current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(
    name='sevq',
    version='1.0.0',
    description='Simplified Evolving Vector Quantization for Classification',
    long_description=long_description,
    long_description_context_type='text/markdown',
    author='Sylwester Czmil',
    author_email='sylwekczmil@gmail.com',
    url='https://github.com/sylwekczmil/sevq',
    packages=['sevq'],
    install_requires=[
        'numpy>=1.18.5'
    ],
)
