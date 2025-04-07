from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nexor",
    version="0.1.0",
    author="letho1608",
    author_email="letho16082003@gmail.com",
    description="A ai library combining the best of TensorFlow and PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/letho1608/nexor",
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
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.9.0",
        "tensorflow>=2.6.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.62.0",
        "rich>=10.0.0",
        "click>=8.0.0"
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'flake8>=3.9.0',
            'black>=21.5b2',
            'isort>=5.8.0',
            'mypy>=0.900',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5.0',
            'sphinx-autodoc-typehints>=1.12.0'
        ]
    },
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        'nexor': ['py.typed'],
    },
    entry_points={
        'console_scripts': [
            'nexor=nexor.cli.main:main',
        ],
    }
)