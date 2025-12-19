<p align="center">
  <img src="./docs/assets/imgs/logo.png" alt="logo" width="600" style="width: 600px;">
</p>

**MEDical TERM EXtraction using Artificial Intelligence.**
This project focuses on developing and fine-tuning models for medical term extraction and general named entity recognition.

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

## 🚀 Quick Start

After setting up your environment and installing the desired framework (GLiNER, Unsloth, or Ollama), you can quickly get started with model training and evaluation.

### Choosing a Model

The project supports three different approaches for medical term extraction:

- **[GLiNER](./docs/models/gliner.md)**: Lightweight NER model, fast training, good for entity extraction with predefined labels
- **[Unsloth](./docs/models/unsloth.md)**: LLM fine-tuning with LoRA, flexible instruction-following, best for complex medical text understanding
- **[Ollama](./docs/models/ollama.md)**: Pre-trained model evaluation only, no training required, good for quick testing

For detailed comparisons and use cases, see the [model documentation](./docs/models).

### Example: Training GLiNER

```bash
# Activate GLiNER environment
source .venv-gliner/bin/activate

# Run training and evaluation script
bash scripts/models/train_eval_model_gliner.sh
```

### Example: Training with Unsloth

```bash
# Activate Unsloth environment
source .venv-unsloth/bin/activate

# Run training and evaluation script
bash scripts/models/train_eval_model_unsloth.sh
```

### Example: Evaluating with Ollama

```bash
# Make sure Ollama service is running
ollama serve

# Run evaluation script
bash scripts/models/eval_model_ollama.sh
```

For more detailed instructions, see the respective model documentation in [./docs/models](./docs/models).

## 🖥️ Running Scripts

Documentation of the different supporting models is available in [./docs/models](./docs/models).

### Python Execution

The project supports both [uv] and standard [python]/[pip] workflows. All scripts automatically detect which is available:

- **[uv]**: Fast Python package and script execution (used if installed, handles venv automatically)
- **[python]**: Standard Python interpreter (always works, scripts auto-activate the appropriate venv)

**Running bash scripts** (recommended):
```bash
# Scripts automatically detect uv or python and handle virtual environments
bash scripts/models/train_eval_model_gliner.sh
bash scripts/models/train_eval_model_unsloth.sh
bash scripts/models/eval_model_ollama.sh
```

**Running Python modules directly**:
```bash
# Using uv (automatically manages virtual environment)
uv run python -m src.training.train_gliner --args...

# Using standard python (activate virtual environment first)
source .venv/bin/activate  # or .venv-gliner/.venv-unsloth depending on the model
python -m src.training.train_gliner --args...
```

> **Note:**
> - When using **bash scripts**, virtual environment activation is handled automatically
> - When using **uv**, no manual venv activation is needed
> - When running Python directly without uv, you must activate the appropriate venv first
> - GLiNER and Unsloth require separate virtual environments due to incompatible dependencies

### Running on SLURM Clusters

For running jobs on HPC clusters with SLURM, see the [SLURM documentation](./SLURM.md) for detailed information about job scheduling and resource management. SLURM-ready scripts are available in the [`slurm/`](./slurm) directory:

- `slurm/train_eval_model_gliner.sh` - GLiNER training and evaluation
- `slurm/train_eval_model_unsloth.sh` - Unsloth training and evaluation

These scripts include all necessary SLURM directives for GPU allocation, resource requests, and job management.

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
