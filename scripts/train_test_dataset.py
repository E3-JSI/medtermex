import os
import re
import json
import random
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset
from itertools import groupby
from argparse import ArgumentParser

def main(hparams):
    dataset_path = hparams.dataset
    train_save = hparams.train_save
    test_save = hparams.test_save

    file_list = os.listdir(dataset_path)

    # Retrive txt and ann files
    txt_files = [file for file in file_list if file.endswith('.txt')]
    ann_files = [file for file in file_list if file.endswith('.ann')]

    # Tuple together same txt and ann files
    file_tuples = [(txt, txt[:-4] + '.ann') for txt in txt_files if txt[:-4] + '.ann' in ann_files]

    mySet = {
        'Diagnostic_procedure',
        'Sign_symptom',
        'Biological_structure',
        'Detailed_description',
        'Age',
        'Lab_value',
        'Sex',
    }

    # Shuffle our dataset
    shuffled_file_tuples = random.sample(file_tuples, k=len(file_tuples))

    train_file_tuples = shuffled_file_tuples[:180]
    test_file_tuples = shuffled_file_tuples[180:]


    # TRAIN DATASET
    df = pd.DataFrame(columns=['input', 'output', 'keys'])

    for x in range(10):
        for txtAnnPair in train_file_tuples:
            pathToTxtFile = os.path.join(dataset_path, txtAnnPair[0])
            pathToAnnFile = os.path.join(dataset_path, txtAnnPair[1])

            Txtfile = open(pathToTxtFile, "r")
            Txtfile = Txtfile.readlines()
            Txtfile = "".join(Txtfile)

            annfile = open(pathToAnnFile, "r")
            allAnnLines = [re.split(r'\t+', tag.rstrip('\t')) for tag in annfile if tag[0][0].startswith(('T'))]
    
            annotations = []
            
            # Retrieving relevant entities defined in mySet
            for annotation in allAnnLines:
                if annotation[1].split()[2].isdigit():
                    CurrentLabel = annotation[1].split()[0]
                    Text = annotation[2].strip()
                    if CurrentLabel not in mySet:
                        continue
                    else:
                        annotations.append((CurrentLabel, Text))

            annDict = {}

            for key, group in groupby(sorted(annotations), lambda x: x[0]):
                listOfThings = ",".join([thing[1] for thing in group])
                annDict[key] = listOfThings.split(",")
            
            # Randomly selecting from 1 to max 5 entities (our keys).
            number = random.randint(1,5)
            keys = random.sample(list(annDict), number)

            # Adding to dataframe (input (raw text), output (correct json), entities to extract (our keys divided by "," and ending with "."))
            df.loc[len(df.index)] = [Txtfile, json.dumps(dict((k, annDict[k]) for k in keys if k in annDict), ensure_ascii=False), ", ".join(keys) + "."]

    # Saving TRAIN DATASET
    hg_dataset = Dataset.from_pandas(df)
    hg_dataset = hg_dataset.remove_columns(['__index_level_0__'])
    hg_dataset.save_to_disk(train_save)

    print("Saving TRAIN dataset")


    # TEST DATASET
    df = pd.DataFrame(columns=['input', 'output', 'keys'])

    for txtAnnPair in test_file_tuples:
        pathToTxtFile = os.path.join(dataset_path, txtAnnPair[0])
        pathToAnnFile = os.path.join(dataset_path, txtAnnPair[1])

        Txtfile = open(pathToTxtFile, "r")
        Txtfile = Txtfile.readlines()
        Txtfile = "".join(Txtfile)

        annfile = open(pathToAnnFile, "r")
        allAnnLines = [re.split(r'\t+', tag.rstrip('\t')) for tag in annfile if tag[0][0].startswith(('T'))]

        annotations = []
        
        # Retrieving relevant entities defined in mySet
        for annotation in allAnnLines:
            if annotation[1].split()[2].isdigit():
                CurrentLabel = annotation[1].split()[0]
                Text = annotation[2].strip()
                if CurrentLabel not in mySet:
                    continue
                else:
                    annotations.append((CurrentLabel, Text))

        annDict = {}

        for key, group in groupby(sorted(annotations), lambda x: x[0]):
            listOfThings = ",".join([thing[1] for thing in group])
            annDict[key] = listOfThings.split(",")

        keys = list(annDict)

        df.loc[len(df.index)] = [Txtfile, dict(annDict), keys]

    # Saving TEST DATASET
    hg_dataset = Dataset.from_pandas(df)
    hg_dataset = hg_dataset.remove_columns(['__index_level_0__'])
    hg_dataset.save_to_disk(test_save)

    print("Saving TEST dataset")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="path to dataset directory")
    parser.add_argument("--train_save", type=str, default=None, help="path to saving training dataset (name it too)")
    parser.add_argument("--test_save", type=str, default=None, help="path to saving test dataset (name it too)")
    args = parser.parse_args()

    main(args)