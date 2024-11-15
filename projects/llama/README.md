# Llama-based Medical Term Extraction

This project focuses on developing and fine-tuning Llama-based medical term extraction models.

## 📚 Papers

In case you use any of the components for your research, please refer to (and cite) the papers:

TODO: Paper

## ⚙️ Environment Variables

This project uses environment variables to configure the training and testing scripts. The following variables are required:

- `access_token`: The access token for the Hugging Face API.

## ⚗️ Experiments

To run the experiments, run the following commands:

```bash
TODO: Provide scripts for the experiments
```

## Examples of running our scripts

Example of calling training_shuffled.py:

```bash
python scripts/training_shuffled.py \
    --model AdaptLLM/medicine-chat \
    --dataset ./data/datasetTrainShuffled \
    --output ./models/AdaptLLM/medicine-chat \
    --project PREPARE
```

Example of calling test_model.py:

```bash
python scripts/test_model.py \
    --model "AdaptLLM/medicine-chat" \
    --adapter "./models/AdaptLLM/medicine-chat/" \
    --dataset "./data/datasetTest" \
    --output "./results/worktesting"
```

### Results

The results folder contains the experiment

TODO: Provide a list/table of experiment results

## 📦️ Available Models

This project produced the following models:

- TODO: Name and the link to the model

## 🚀 Using the Available Models

When the model is trained, the following script shows how one can use the model:

```python
TODO: Provide an example of how to use the model
```
