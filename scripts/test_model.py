import sys
import json
from collections import defaultdict
from argparse import ArgumentParser

from tqdm import tqdm
from datasets import load_from_disk

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
from guidance import models, gen, select

# TODO: Resolve the scripts problem
sys.path.append("..")
from src.models.generate import Generate

def main(hparams):
    base_model_name = hparams.model
    adapter = hparams.adapter
    dataset = hparams.dataset
    save_results = hparams.output

    f = open(save_results+".txt", "w")

    # Import model
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(base_model_name, load_in_4bit=True)
    model = PeftModel.from_pretrained(model, adapter).to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    # Import test dataset
    test_dataset = load_from_disk(dataset)

    f1_score = defaultdict(list)

    compare = ""
    #entities = ["Age"]
    #entities = ["Age", "Sex", "Sign_symptom", "Lab_value"]
    entities = ["Age", "Sex", "Sign_symptom", "Lab_value", "Biological_structure", "Diagnostic_procedure"]

    for instance in tqdm(range(len(test_dataset['input']))):
        text = test_dataset['input'][instance].replace(" ", " ")
        text = text.replace("℃", "°C")

        extract = Generate.extract(model, tokenizer, entities, text, max_length=2000)

        compare = compare + test_dataset[instance]["input"] + "\n\n"

        for x in json.loads(test_dataset['output'][instance]):

            if x not in entities:
                continue

            if x not in extract:
                text = f"{x} \nReal: {json.loads(test_dataset['output'][instance])[x]} \nGene: Failed \nMatch: \n\n"
                compare = compare + text
                f1_score[x].append(0)
                continue

            if json.loads(test_dataset['output'][instance])[x] is None:
                real = ["None"]
            else:
                real = list(set(json.loads(test_dataset['output'][instance])[x]))

            real = list(set(json.loads(test_dataset['output'][instance])[x]))
            gene = list(set(extract[x]))

            match = list(set([r for r in real if r in gene]))

            lenmatch = len(match)
            lenreal = len(real)
            lengene = len(gene)
                
            text = f"{x} \nReal: {json.loads(test_dataset['output'][instance])[x]} \nGene: {extract[x]} \nMatch: {match} \n\n"
            compare = compare + text

            precision = lenmatch/lengene
            recall = lenmatch/lenreal

            if precision == 0 and recall == 0:
                f1 = 0
            else:
                f1 = 2*(precision*recall) / (precision+recall)

            f1_score[x].append(f1) 
                
        compare = compare + "\n\n\n"
    
    compare = compare + json.dumps(average_f1_dict(f1_score))

    f.write(compare)
    f.close()


def average_f1_dict(di):
    return {k:sum(v)/len(v) for k,v in di.items()}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="model_id (hugging face)")
    parser.add_argument("--adapter", type=str, default=None, help="trained adapter")
    parser.add_argument("--dataset", type=str, default=None, help="path to testing dataset")
    parser.add_argument("--output", type=str, default=None, help="path for saving the results")
    args = parser.parse_args()

    main(args)



