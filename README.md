# Deneir Script

This is a template repository for creating an experiment environment in Python.
It intends to speed up the research process - reducing the repository
structure design - and to have it clean and concise through multiple
experiments.

Inspired by the [cookiecutter][cookiecutter] folder structure.

**Instructions:**

- Search for all TODOs in the project and add the appropriate values
- Rename this README title and description

## ☑️ Requirements

Before starting the project make sure these requirements are available:

- [conda][conda]. For setting up your research environment and Python dependencies.
- [dvc][dvc]. For versioning your data.
- [git][git]. For versioning your code.

## 🛠️ Setup

### Create a python environment

First, create a virtual environment where all the modules will be stored.

#### Using virtualenv

Using the `venv` command, run the following commands:

```bash
# create a new virtual environment
python -m venv venv

# activate the environment (UNIX)
source ./venv/bin/activate

# activate the environment (WINDOWS)
./venv/Scripts/activate

# deactivate the environment (UNIX & WINDOWS)
deactivate
```

#### Using conda

Install [conda], a program for creating Python virtual environments. Then run the following commands:

```bash
# create a new virtual environment
conda create --name [TODO] python=3.8 pip

# activate the environment
conda activate [TODO]

# deactivate the environment
deactivate
```

### Install

To install the requirements run:

```bash
pip install -e .
```

## 🗃️ Data

TODO: Provide information about the data used in the experiments

- Where is the data found
- How is the data structured

## ⚗️ Experiments

To run the experiments, run the following commands:

```bash
TODO: Provide scripts for the experiments
```

### 🦉 Using DVC

An alternative way of running the whole experiment is by using [DVC][dvc]. To do this,
simply run:

```bash
dvc exp run
```

This command will read the `dvc.yaml` file and execute the stages accordingly, taking
any dependencies into consideration.

### Results

The results folder contains the experiment

TODO: Provide a list/table of experiment results

## 📦️ Available models

This project produced the following models:

- TODO: Name and the link to the model

## 🚀 Using the trained model

When the model is trained, the following script shows how one can use the model:

```python
TODO: Provide an example of how to use the model
```

## 📚 Papers

In case you use any of the components for your research, please refer to
(and cite) the papers:

TODO: Paper

### 📓 Related work

TODO: Related paper

## 🚧 Work In Progress

- [ ] Setup script
- [ ] Code for data prep
- [ ] Code for model training
- [ ] Code for model validation
- [ ] Code for model evaluation
- [ ] Modify `params.yaml` and modify the scripts to read the params from the file
- [ ] Modify DVC pipelines for model training and evaluation

## 📣 Acknowledgments

This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

This work is supported by the Slovenian Research Agency and the TODO.

[cookiecutter]: https://drivendata.github.io/cookiecutter-data-science/
[python]: https://www.python.org/
[conda]: https://www.anaconda.com/
[git]: https://git-scm.com/
[dvc]: https://dvc.org/
[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
