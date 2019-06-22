"""
@Author: Kumar Nityan Suman
@Date: 2019-06-23 00:34:33
"""


# Load package
import setuptools

# Read complete description
with open("README.md", mode="r") as fh:
    long_description = fh.read()

# Create setup
setuptools.setup(
    name="tensorhub",
    version="1.0-alpha0",
    author="Kumar Nityan Suman",
    author_email="nityan.suman@gmail.com",
    description="Library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nityansuman/tensorhub",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)