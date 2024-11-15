from setuptools import setup, find_packages

with open("README.md", mode="r") as fh:
    long_description = fh.read()

with open("../../requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name="gliner",
    version="0.0.1",
    author="E3-JSI",
    description="The GLiNER-based Medical Term Extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    install_requires=[
        req.strip() for req in requirements if req.strip() and not req.startswith("#")
    ],
)
