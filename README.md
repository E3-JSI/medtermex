<p align="center">
  <img src="./docs/assets/imgs/logo.png" alt="logo" width="600" style="width: 600px;">
</p>

**Medical Term Extraction using Artificial Intelligence.**
This project focuses on developing and fine-tuning models for medical term extraction.

## 📚 Papers

In case you use any of the components for your research, please refer to (and cite) the papers:

TODO: Paper

## ☑️ Requirements

Before starting the project make sure these requirements are available:

- [python]. For setting up the research environment and Python dependencies (version 3.10 or higher).
- [git]. For versioning your code.

## 📁 Project Structure

The project is structured as follows:

```plaintext
.
├── data/                   # Data used in the experiments
├── common/                 # Common utilities and modules
├── projects/               # The different projects in the repository
│   ├── gliner
│   └── llama
├── results/                # Results of the experiments
├── models/                 # Trained models
├── .gitignore              # Files and directories to be ignored by git
├── README.md               # The main README file
├── requirements-dev.txt    # Development dependencies
├── requirements.txt        # Project dependencies
└── setup.py                # Setup script
```

## 🛠️ Setup

### Create a python environment

First, create a virtual environment where all the modules will be stored.

#### Using virtualenv

Using the `venv` command, run the following commands:

```bash
# create a new virtual environment
python -m venv venv

# activate the environment (UNIX)
. ./venv/bin/activate

# activate the environment (WINDOWS)
./venv/Scripts/activate

# deactivate the environment (UNIX & WINDOWS)
deactivate
```

### Install

Since the project is a monorepo, we will install the dependencies in the following way:

```bash
# install the general dependencies
pip install -e .[dev]
```

> [!NOTE]
> The `dev` extra installs the development dependencies.

Next, install the dependencies for the projects:

```bash
# install all projects' dependencies
pip install -e projects/*
```

> [!NOTE]
> You can install separate projects by running `pip install -e projects/<project_name>`.
> See the [Projects](#-Projects) section for more information.

#### Adding a new project

To add a new project, add the project to the `projects` directory and install the project dependencies.

```bash
pip install -e projects/<project_name>
```

### Install the pre-commit hooks

To install the pre-commit hooks, run the following command:

```bash
pre-commit install
```

## ⚙️ Environment Variables

Some of the projects require environment variables to be set (see individual projects for details). To set the environment variables, copy the `.env.example` file to `.env` and replace the values with the correct ones.

## 🚀 Projects

The repository contains multiple projects, each associated with their own way of
extracting medical terms. Each project is located in the `projects` directory.

Currently, the following projects are available:

- [gliner](projects/gliner/README.md). Medical term extraction using the GLiNER models.
- [llama](projects/llama/README.md). Medical term extraction using the Llama models.

## 🗃️ Data

TODO: Provide information about the data used in the experiments

- Where is the data found
- How is the data structured

## 🧪 Tests

To run existing tests, use the following command:

```bash
python -m unittest discover test*
```

## 🧹 Cleanup

To cleanup the project, remove the virtual environment and generated files.

```bash
# deactivate the environment
deactivate
# remove the virtual environment
rm -rf venv
# remove the generated files
find . -type d -name '*.egg-info' -exec rm -rf {} +
```

## 📣 Acknowledgments

This work is developed by the [Department of Artificial Intelligence][ailab]
at [Jozef Stefan Institute][ijs], and other contributors.

This work is supported by the Slovenian Research Agency.
The project has received funding from the European Union's Horizon Europe research
and innovation programme under [[Grant No. 101080288][grant]] ([PREPARE]).

<figure>
  <img src="./docs/assets/imgs/EU.png?raw=true" alt=European Union flag" width="80" />
</figure>

[python]: https://www.python.org/
[git]: https://git-scm.com/
[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
[PREPARE]: https://prepare-rehab.eu/
[grant]: https://cordis.europa.eu/project/id/101080288
