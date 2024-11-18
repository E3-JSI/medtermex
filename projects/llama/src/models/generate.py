import json
import warnings
from typing import Any, List

from guidance import gen
from src.models.input_prompts import Prompts

warnings.filterwarnings("ignore")

# ================================================
# Helper functions
# ================================================


def format_json_format(pred: str):
    # TODO: Solve possible issue with nested JSONs
    sep = "}"
    pred = pred.replace("'", '"')
    pred = pred.split(sep, 1)[0] + sep
    pred = pred.replace("\\", " ")

    try:
        json_pred = json.loads(pred)
    except json.JSONDecodeError:
        print("Unable to parse JSON")
        return {}
    else:
        return json_pred


# ================================================
# Main classes
# ================================================


class Generate:
    @classmethod
    def generate(
        cls, model: Any, tokenizer: Any, prompt: str, max_length: int = 2000
    ) -> str:
        inputs = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        outputs = model.generate(
            input_ids=inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id
        )[0]
        answer_start = int(inputs.shape[-1])
        pred = tokenizer.decode(outputs[answer_start:], skip_special_tokens=True)
        return pred

    @classmethod
    def extract(
        cls,
        model: Any,
        tokenizer: Any,
        entities: List[str],
        text: str,
        max_length: int = 2000,
    ) -> dict:
        output = {}
        for entity in entities:
            try:
                prompt = getattr(Prompts, entity.lower())(text)
                json_pred = format_json_format(
                    cls.generate(model, tokenizer, prompt, max_length)
                )
                output.update(json_pred)
            except AttributeError as e:
                print(f"AttributeError: {e}")
                continue
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue
        return output


class GenerateWithGuidance:
    @classmethod
    def generate(cls, model: Any, prompt: str, max_tokens: int) -> str:
        lm = model + prompt
        lm += "\n" + gen("pred", max_tokens=max_tokens, stop="}") + "}"
        return lm["pred"]

    @classmethod
    def extract(
        cls, model: Any, entities: List[str], text: str, max_length: int
    ) -> dict:
        output = {}
        for entity in entities:
            try:
                prompt = getattr(Prompts, entity.lower())(text)
                output.update(
                    format_json_format(cls.generate(model, prompt, max_length))
                )
            except Exception as e:
                print(e)
                continue
        return output
