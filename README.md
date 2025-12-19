<p align="center">
  <img src="./docs/assets/imgs/logo.png" alt="logo" width="600" style="width: 600px;">
</p>

**Medical Term Extraction using Artificial Intelligence.**
This project focuses on developing and fine-tuning models for medical term extraction.

The project currently supports GLiNER, LLMs (using Unsloth) and Ollama models. It includes scripts for fine-tuning using LoRA, and provides examples for fine-tuning the models both locally and on [SLURM].

> **Note:** GLiNER and Unsloth have incompatible dependency requirements and must be installed in separate virtual environments. See the [Installation Options](#installation-options) section for details.

## ☑️ Requirements

Before starting the project make sure these requirements are available:

- [uv] or [python] (version 3.10 or higher). For setting up the environment and Python dependencies.
- [git]. For versioning your code.

## 📁 Project Structure

The project is structured as follows:

```plaintext
.
├── data/                   # Data used in the experiments
│   ├── raw/                # Raw data
│   ├── interim/            # Intermediate data
│   ├── final/              # Final processed data
│   ├── external/           # External data
│   └── README.md           # Data documentation
├── src/                    # Source code
│   ├── core/               # Core modules and utilities
│   ├── pipelines/          # Data and processing pipelines
│   └── training/           # Training modules
├── scripts/                # Utility scripts
├── docs/                   # Documentation
├── results/                # Results of the experiments
├── models/                 # Trained models
├── logs/                   # Log files
├── slurm/                  # SLURM job scripts
├── .gitignore              # Files and directories to be ignored by git
├── README.md               # The main README file
├── Makefile                # Make targets for setup, cleanup, and linting
├── pyproject.toml          # Project configuration and dependencies
├── setup.cfg               # Setup configuration
├── .python-version         # Python version specification
├── CHANGELOG.md            # Project changelog
├── LICENSE                 # Project license
└── SLURM.md                # SLURM documentation
```

## 🛠️ Setup

### Python version

The Python version for this project is specified in the `.python-version` file. This file should contain only the major and minor version number (e.g., `3.12`).

If the `.python-version` file is not present or contains an invalid format, the setup script will default to Python version installed on the machine.

To change the Python version:

1. Create and/or edit the `.python-version` file in project root
2. Specify the desired version in `X.Y` format (e.g., `3.10`, `3.11`, `3.12`, `3.13`)
3. Re-run the setup process (see below)

### Setup the environment

To set up the development environment, run the following command:

```bash
make setup
```

This will:

- Create a virtual environment at `.venv`
- Install core project dependencies (using `uv` if available, otherwise `pip`)
- Create necessary data directories (`data/raw`, `data/interim`, `data/final`, `data/external`)

> [!NOTE]
> The Python version is specified in `.python-version`. The setup script will use this version automatically.

> [!NOTE]
> The `make setup` command installs only the core dependencies. To use GLiNER or Unsloth, you must install them separately as optional dependencies (see [Installation Options](#installation-options)).

### Installation Options

The project supports multiple ML frameworks as optional dependencies. You can install only the frameworks you need:

```bash
# Install with specific framework
pip install -e .[gliner]    # For GLiNER models
pip install -e .[unsloth]   # For LLM fine-tuning with Unsloth
pip install -e .[ollama]    # For Ollama models

# Install all dependencies for a specific framework
pip install -e .[all-gliner]   # GLiNER + Ollama + dev tools
pip install -e .[all-unsloth]  # Unsloth + Ollama + dev tools

# Install only core dependencies (no framework-specific packages)
pip install -e .

# Install with development tools
pip install -e .[dev]
```

> [!WARNING]
> **GLiNER and Unsloth are incompatible and cannot be installed together!**
>
> - GLiNER requires: `transformers>=4.38.2,<=4.51.0`
> - Unsloth requires: `transformers>=4.51.3`
>
> These version ranges do not overlap. To use both frameworks, you must create **separate virtual environments**:
>
> ```bash
> # Environment for GLiNER
> python -m venv .venv-gliner
> source .venv-gliner/bin/activate
> pip install -e ".[gliner]"
>
> # Environment for Unsloth (create separately)
> python -m venv .venv-unsloth
> source .venv-unsloth/bin/activate
> pip install -e ".[unsloth]"
> ```

**Framework-specific dependencies:**
- `[gliner]`: GLiNER model training and evaluation
- `[unsloth]`: LLM fine-tuning with LoRA using Unsloth
- `[ollama]`: Ollama model integration
- `[dev]`: Development tools (black, isort, flake8, pre-commit)
- `[all-gliner]`: GLiNER + Ollama + dev tools (use for GLiNER projects)
- `[all-unsloth]`: Unsloth + Ollama + dev tools (use for Unsloth projects)

## ⚙️ Environment Variables

Some components may require environment variables to be set. To set the environment variables, copy the `.env.example` file (if available) to `.env` and replace the values with the correct ones.

## 🚀 Running Scripts

Documentation of the different supporting models is available in [./docs/models](./docs/models).

Scripts and experiments in this project are run using:

- [uv] (default, if available): Fast Python script execution
- [python] (fallback): Regular Python interpreter

Both are supported. When [uv] is available, it will automatically be used for faster execution. You can explicitly use either:

```bash
# Using uv (faster)
uv run script_name.py

# Using python (always available)
python script_name.py
```

## 🧹 Cleanup

To clean up the project, run the following command:

```bash
make cleanup
```

This will remove generated files, caches, and compiled Python files.

## 📣 Acknowledgments

This work is developed by the [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs], and other contributors.

This work is supported by the Slovenian Research Agency.
The project has received funding from the European Union's Horizon Europe research
and innovation programme under [[Grant No. 101080288][PREPARE-GRANT]] ([PREPARE]).

<figure>
  <img src="./docs/assets/imgs/EU.png?raw=true" alt=European Union flag" width="80" />
</figure>

[SLURM]: https://slurm.schedmd.com/documentation.html

[uv]: https://docs.astral.sh/uv/
[python]: https://www.python.org/
[git]: https://git-scm.com/

[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
[PREPARE]: https://prepare-rehab.eu/
[PREPARE-GRANT]: https://cordis.europa.eu/project/id/101080288
