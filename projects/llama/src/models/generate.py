
import warnings
import json
from src.models.input_prompts import Prompts

warnings.filterwarnings("ignore")

class Generate:
    def gene(model, tokenizer, prompt, max_length=2000):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        outputs = model.generate(input_ids=inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)[0]
        answer_start = int(inputs.shape[-1])
        pred = tokenizer.decode(outputs[answer_start:], skip_special_tokens=True)
        return pred
    
    def json_format(pred):
        # TODO: Solve possible issue with nested JSONs
        pred = pred.replace("\'", "\"")
        sep = "}"
        pred = pred.split(sep, 1)[0] + sep
        pred = pred.replace("\\"," ")
        try:
            json_pred = json.loads(pred)
        except:
            print("Something went wrong with json.loads(pred)")
            print(pred)
            return {}
        else:
            json_pred = json.loads(pred)
            return json_pred
    
    def extract(model, tokenizer, entities, text, max_length=2000):
        output = {}
        for entity in entities:
            try:
                prompt = getattr(Prompts, entity.lower())(text)
                json_pred = Generate.json_format(Generate.gene(model, tokenizer, prompt, max_length))
                output.update(json_pred)
            except Exception as e:
                print(e)
                continue
        return output
