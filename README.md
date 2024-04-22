# Medical-LLM

This project focuses on developing and fine-tuning LLMs for medical tasks.

Inspired by the [cookiecutter] folder structure.

## ☑️ Requirements

Before starting the project make sure these requirements are available:

- [python]. For setting up the research environment and Python dependencies (version 3.8 or higher).
- [git]. For versioning your code.

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

### Install

To install the requirements run:

```bash
pip install -e .
```

## 🗃️ Data

TODO: Provide information about the data used in the experiments

- Where is the data found
- How is the data structured

## Create the Environment File
Create a copy of the `.env.example` file named `.env` and replace the `access_token` value with the access token to your Hugging Face account.

## ⚗️ Experiments

To run the experiments, run the following commands:

```bash
TODO: Provide scripts for the experiments
```
## Examples of running our scripts

Example of calling training_shuffled.py: 
python scripts/training_shuffled.py \
    --model AdaptLLM/medicine-chat \
    --dataset ./data/datasetTrainShuffled \
    --output ./models/AdaptLLM/medicine-chat \
    --project PREPARE

Example of calling test_model.py: 
python scripts/test_model.py \
    --model "AdaptLLM/medicine-chat" \
    --adapter "./models/AdaptLLM/medicine-chat/" \
    --dataset "./data/datasetTest" \
    --output "./results/worktesting"

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

In case you use any of the components for your research, please refer to (and cite) the papers:

TODO: Paper

### 📓 Related work

#### Models

[Chen, Zeming, et al. "MEDITRON-70B: Scaling Medical Pretraining for Large Language Models." arXiv preprint arXiv:2311.16079 (2023).](https://arxiv.org/abs/2311.16079)

[Wang, Guangyu, et al. "ClinicalGPT: Large Language Models Finetuned with Diverse Medical Data and Comprehensive Evaluation." arXiv preprint arXiv:2306.09968 (2023).](https://arxiv.org/abs/2306.09968)

[Wu, Chaoyi, et al. "Pmc-llama: Further finetuning llama on medical papers." arXiv preprint arXiv:2304.14454 (2023).](https://arxiv.org/abs/2304.14454)

[Touvron, Hugo, et al. "Llama 2: Open foundation and fine-tuned chat models." arXiv preprint arXiv:2307.09288 (2023).](https://arxiv.org/abs/2307.09288)

[Jiang, Albert Q., et al. "Mistral 7B." arXiv preprint arXiv:2310.06825 (2023).](https://arxiv.org/abs/2310.06825)

#### Optimization Algorithms

[Dettmers, Tim, et al. "Qlora: Efficient finetuning of quantized llms." arXiv preprint arXiv:2305.14314 (2023).](https://arxiv.org/abs/2305.14314)

## 🚧 Work In Progress

- [ ] Code for data preparation
- [ ] Code for model training
- [ ] Code for model validation
- [ ] Code for model evaluation

## 📣 Acknowledgments

This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

This work is supported by the Slovenian Research Agency and the Horizon Europe [PREPARE] project [[Grant No. 101080288][grant]].

[cookiecutter]: https://drivendata.github.io/cookiecutter-data-science/
[python]: https://www.python.org/
[git]: https://git-scm.com/
[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
[PREPARE]: https://prepare-rehab.eu/
[grant]: https://cordis.europa.eu/project/id/101080288
