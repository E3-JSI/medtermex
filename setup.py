from setuptools import find_namespace_packages, setup

with open("README.md", mode="r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

with open("requirements-dev.txt", "r") as fh:
    requirements_dev = fh.readlines()

setup(
    name="monorepo",
    version="0.0.1",
    author="E3-JSI",
    description="The monorepo project setup",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["projects.*"]),
    install_requires=[
        req.strip() for req in requirements if req.strip() and not req.startswith("#")
    ],
    extras_require={
        "dev": [
            req.strip()
            for req in requirements_dev
            if req.strip() and not req.startswith("#")
        ]
    },
)
