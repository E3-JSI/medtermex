import json
import os

class Prompts:
    def __init__(self):
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(CURRENT_DIR, "prompts.json")

        with open(prompt_path, "r") as file:
            self.data = json.load(file)

    def _get_description(self, label, output_type="few_shot_prompting"):
        return self.data[output_type]["labels"][label]["description"]

    def _get_output_structure(self, label, output_type="few_shot_prompting"):
        return self.data[output_type]["labels"][label]["output_structure"]

    def _content(self, description, output_structure):
        output_structure = json.dumps(output_structure)
        description_prompt = "Please extract " + description.lower() + " Answer must follow the following format:\n[" + output_structure + "...]"
        return description_prompt

    def _get_system_prompt(self, type):
        return self.data[type]["system_prompt"]

    def create_prompt_only_prompt(self, label, medical_text):
        """Returns the messages for prompt only
    
            Args:
                label: string - The label to be extracted.
                medical_text: string - The medical text the label is extracted.

            Returns:
                List[dict]: A list of system, user messages.

        """
        message = []
        description = self._get_description(label, output_type="prompt_only")
        output_structure = self._get_output_structure(label, output_type="prompt_only")
        message.append({"role": "system", "content": self._get_system_prompt("prompt_only")})
        message.append({"role": "user", "content": self._content(description, output_structure) + "\n\nMedical text:\n" + medical_text})
        return message

    def _get_example(self, label, example_number):
        return self.data["few_shot_prompting"]["labels"][label]["example_" + str(example_number)]

    def create_few_shot_prompt(self, label, medical_text):
        """Returns the messages for few shot prompting with 3 examples.
    
            Args:
                label: string - The label to be extracted.
                medical_text: string - The medical text the label is extracted.

            Returns:
                List[dict]: A list of system, user, assistant messages.

        """
        message = []
        description = self._get_description(label, output_type="few_shot_prompting")
        output_structure = self._get_output_structure(label, output_type="few_shot_prompting")
        message.append({"role": "system", "content": self._get_system_prompt("prompt_only")})
        message.append({"role": "user", "content": self._content(description, output_structure) + "\n\nMedical text:\n" + self._get_example(label, 1)["input"]})
        message.append({"role": "assistant", "content": self._get_example(label, 1)["output"]})
        message.append({"role": "user", "content": self._content(description, output_structure) + "\n\nMedical text:\n" + self._get_example(label, 4)["input"]})
        message.append({"role": "assistant", "content": self._get_example(label, 4)["output"]})
        message.append({"role": "user", "content": self._content(description, output_structure) + "\n\nMedical text:\n" + self._get_example(label, 2)["input"]})
        message.append({"role": "assistant", "content": self._get_example(label, 2)["output"]})
        message.append({"role": "user", "content": self._content(description, output_structure) + "\n\nMedical text:\n" + medical_text})
        return message
    
    def _create_conversational_prompt(self, labels, output_structure, medical_text):
        output_structure = json.dumps(output_structure)
        labels_string = ', '.join(str(label) for label in labels)
        instruction_content = "Please extract the following entities: " + labels_string + ". Answer must follow the following format:\n[" + output_structure + "...]"
        prompt = instruction_content + "\n\nMedical text:\n" + medical_text
        return prompt
    
    def create_conversational_training_message_with_completion(self, labels, medical_text, system_output):
        # Function is for training the model on the conversational prompt
        message = []
        output_structure = self._get_output_structure("no_label", output_type="conversational_training")
        message.append({"role": "system", "content": self._get_system_prompt("conversational_training")})
        message.append({"role": "user", "content": self._create_conversational_prompt(labels, output_structure, medical_text)})
        message.append({"role": "assistant", "content": json.dumps(system_output)})
        return message
    
    def create_conversational_message(self, labels, medical_text):
        # Function is for evaluating the model on the conversational prompt
        message = []
        output_structure = self._get_output_structure("no_label", output_type="conversational_training")
        message.append({"role": "system", "content": self._get_system_prompt("conversational_training")})
        message.append({"role": "user", "content": self._create_conversational_prompt(labels, output_structure, medical_text)})
        return message
    
    def _create_instruction_prompt(self, labels, output_structure, medical_text):
        output_structure = json.dumps(output_structure)
        labels_string = ', '.join(str(label) for label in labels)
        instruction_content = "Please extract the following entities: " + labels_string + ". Answer must follow the following format:\n[" + output_structure + "...]"
        prompt = self._get_system_prompt("instruction_training") + "\n\n" + instruction_content + "\n\nMedical text:\n" + medical_text
        return prompt
    
    def create_instruction_training_message_with_completion(self, labels, medical_text, system_output):
        # Function is for training the model on the instruction prompt
        message = {}
        output_structure = self._get_output_structure("no_label", output_type="instruction_training")
        message["prompt"] = self._create_instruction_prompt(labels, output_structure, medical_text)
        message["completion"] = json.dumps(system_output)
        return message
    
    def create_instruction_message(self, labels, medical_text):
        # Function is for evaluating the model on the instruction prompt
        message = {}
        output_structure = self._get_output_structure("no_label", output_type="instruction_training")
        message["prompt"] = self._create_instruction_prompt(labels, output_structure, medical_text)
        return message
    
    