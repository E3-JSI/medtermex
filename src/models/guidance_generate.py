import warnings
import json

from src.models.input_prompts import Prompts
from guidance import models, gen, select

warnings.filterwarnings("ignore")

class Generate:
    def gene(model, prompt, max_tokens):
        lm = model + prompt
        lm += "\n" + gen("pred", max_tokens=max_tokens, stop="}") + "}"
        return lm["pred"]

    def json_format(pred):
        # TODO: Solve possible issue with nested JSONs
        sep = "}"
        pred = pred.split(sep, 1)[0] + sep
        try:
            json_pred = json.loads(pred)
        except:
            print("Something went wrong with json.loads(pred)")
            return {}
        else:
            json_pred = json.loads(pred)
            return json_pred

    def extract(model, entities, text, max_length):
        output = {}
        for entity in entities:
            try:
                prompt = getattr(Prompts, entity.lower())(text)
                output.update(Generate.json_format(Generate.gene(model, prompt, max_length)))
            except Exception as e:
                print(e)
                continue
        return output