# GLiNER-based Medical Term Extraction

This project focuses on developing and fine-tuning LLMs GLiNER-based medical term extraction models.

## 📚 Papers

In case you use any of the components for your research, please refer to (and cite) the papers:

TODO: Paper

## 📁 Data

To train or fine-tune the models, the following datasets are used:

### PubMed

The pubmed dataset contains examples from PubMed. Each example consists of texts and named entity labels associated with medical terms. While the dataset is publicly available, the data is not included in this repository. To download the dataset, run the following command:

```bash
TODO: Provide the download link
```

#### Preprocessing

To process the dataset, run the following commands:

**Format the raw data to JSON**

```bash
python projects/gliner/scripts/pubmed/01_format_raw_to_json.py \
    --input_dir ./data/raw/maccrobat/MACCROBAT2018 \
    --output_dir ./data/interim/MACCROBAT2018 \
    --train_test_ratio 0.8
```

**Format the JSON data to GLiNER train format**

```bash
python projects/gliner/scripts/pubmed/02_format_json_to_gliner_train.py \
    --input_file ./data/interim/MACCROBAT2018/train.json \
    --output_file ./data/final/MACCROBAT2018/train.gliner.json \
    --model_name urchade/gliner_large_bio-v0.1
```

**Format the JSON data to GLiNER test format**

```bash
python projects/gliner/scripts/pubmed/02_format_json_to_gliner_test.py \
    --input_file ./data/interim/MACCROBAT2018/test.json \
    --output_file ./data/final/MACCROBAT2018/test.gliner.json
```

## ⚗️ Experiments

To run the experiments, run the following commands:

**Train the GLiNER models**

```bash
python projects/gliner/scripts/train_model.py \
    --data_train_file ./data/final/MACCROBAT2020/train.gliner.json \
    --model_name_or_path urchade/gliner_large-v2.1 \
    --model_output_dir ./models/gliner_large-v2.1 \
    --num_train_epochs 3 \
    --batch_size 8 \
    --learning_rate 5e-6
```

**Evaluate the GLiNER models**

```bash
python projects/gliner/scripts/evaluate_model.py \
    --data_test_file ./data/final/MACCROBAT2020/test.gliner.json \
    --output_file ./results/MACCROBAT2020/gliner_large-v2.1.json \
    --model_name_or_path ./models/gliner_large-v2.1 \
    --threshold 0.5
```

### Results

The results folder contains the experiment results. The results are stored in JSON format.

We formatted and present the results as a [Google Sheet] document.

## 📦️ Available Models

This project produced the following models:

- TODO: Name and the link to the model

## 🚀 Using the Available Models

When the model is trained, the following script shows how one can use the model:

```python
from gliner import GLiNER

# load the local model
model = GLiNER.from_pretrained(
    "path/to/the/model", load_tokenizer=True, local_files_only=True
)

# (optional) load the model on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# predict the named entities
text = "The patient was diagnosed with a severe case of pneumonia."
labels = ["Sign symptom"]
threshold = 0.5

predictions = model.predict_entities(text, labels, threshold=threshold)

# print the predictions
for p in predictions:
    print(f"{p['text']} => {p['label']}")

```

### Deploying the model to HuggingFace

To deploy the model to HuggingFace, run the following script in python:

```python
from gliner import GLiNER

# load the local model
model = GLiNER.from_pretrained("path/to/the/model")

# deploy the model to HuggingFace
model.push_to_hub("your-hf-username/your-model-name")

# configure the model from HuggingFace ...
```

[Google Sheet]: https://docs.google.com/spreadsheets/d/1-OtsyZ2XjnY6GUHPVQEwDo6pWqXB3QZ0Ec2AH0Y5W2s/edit?usp=sharing
