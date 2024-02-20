from setuptools import setup, find_packages

setup(
    name='sfs-shapwise-feature-selection',
    version='0.1.0',
    author='Roni Goldshmidt',
    author_email='roni.goldsmid@gmail.com',
    description='A Python package for feature selection using SHAP values.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ronigold/SFS-Shapwise-Feature-Selection',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
