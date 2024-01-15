from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="texor",
    version="1.0.0",
    author="letho1608",
    author_email="letho16082003@gmail.com",
    description="Lightweight native deep learning framework with PyTorch-style API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/letho1608/texor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    install_requires=[
        "numpy>=1.19.0",
        "numba>=0.56.0",
        "rich>=10.0.0",
        "click>=8.0.0"
    ],
    extras_require={
        'gpu': [
            'cupy-cuda11x>=9.0.0',  # GPU support
        ],
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.5b2',
            'isort>=5.8.0',
            'mypy>=0.900',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5.0',
        ],
        'viz': [
            'matplotlib>=3.3.0',
            'tqdm>=4.62.0',
        ]
    },
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        'texor': ['py.typed'],
    },
    entry_points={
        'console_scripts': [
            'texor=texor.cli.main:main',
        ],
    },
    keywords=['machine-learning', 'deep-learning', 'neural-networks', 'pytorch', 'numpy', 'autograd'],
)