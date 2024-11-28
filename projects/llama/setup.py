from setuptools import find_packages, setup

with open("README.md", mode="r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name="mte_llama",
    version="0.0.1",
    author="E3-JSI",
    description="The Llama-based Medical Term Extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    install_requires=[
        req.strip() for req in requirements if req.strip() and not req.startswith("#")
    ],
)
