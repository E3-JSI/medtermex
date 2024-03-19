from argparse import ArgumentParser
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from datasets import load_from_disk
import sys
sys.path.append("..")
from scripts.generate import generate
from collections import defaultdict


# EXAMPLE:  python test.py --model "AdaptLLM/medicine-chat" --adapter "../models/AdaptLLM/medicine-chat-Erik-shuffled/" --dataset "../data/datasetTestShuffledEntire" --output "../results/worktesting2"
# EXAMPLE Mistral:  python test.py --model "BioMistral/BioMistral-7B" --adapter "../models/BioMistral/Medicine-chat-Erik-shuffled/" --dataset "../data/datasetTestShuffledEntire" --output "../results/worktesting2"
def main(hparams):
    base_model_name = hparams.model
    adapter = hparams.adapter
    dataset = hparams.dataset
    save_results = hparams.output

    f = open(save_results+".txt", "w")

    #import model
    device = "cuda"
    #base_model_name = "AdaptLLM/medicine-chat" #path/to/your/model/or/name/on/hub"
    #base_model_name = "BioMistral/BioMistral-7B"
    #adapter_model_name = "../models/AdaptLLM/medicine-chat-Erik-shuffled"

    model = AutoModelForCausalLM.from_pretrained(base_model_name, load_in_4bit=True)
    model = PeftModel.from_pretrained(model, adapter).to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    #import test dataset
    test_dataset = load_from_disk(dataset)

    F1score = defaultdict(list)

    compare = ""
    #entities = ["Age"]
    #entities = ["Age", "Sex", "Sign_symptom", "Lab_value"]
    entities = ["Age", "Sex", "Sign_symptom", "Lab_value", "Biological_structure", "Diagnostic_procedure"]

    for instance in tqdm(range(len(test_dataset['input']))):
        text = test_dataset['input'][instance]
        #print(text)

        extract = generate.extract(model, tokenizer, entities, text, max_length=2000)
        #print(extract)

        compare = compare + test_dataset[instance]["input"] + "\n\n"

        for x in json.loads(test_dataset['output'][instance]):

            if x not in entities:
                continue

            if x not in extract:
                text = f"{x} \nReal: {json.loads(test_dataset['output'][instance])[x]} \nGene: Failed \nMatch: \n\n"
                compare = compare + text
                F1score[x].append(0)
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

            #f1 = 2*(precision*recall) / (precision+recall)
            #print(precision, recall, f1)

            F1score[x].append(f1) 
                
                
        compare = compare + "\n\n\n"
    
    
    compare = compare + json.dumps(averageF1Dict(F1score))

    
    f.write(compare)
    f.close()


def averageF1Dict(di):
    return {k:sum(v)/len(v) for k,v in di.items()}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None) #--> model_id (hugging face)
    parser.add_argument("--adapter", type=str, default=None) #--> trained adapter
    parser.add_argument("--dataset", type=str, default=None) #--> path to testing dataset
    parser.add_argument("--output", type=str, default=None) #--> path for saving the results
    args = parser.parse_args()

    main(args)



