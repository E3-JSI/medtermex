import json
import os

# Load the JSON file
prompt_path = os.getcwd()+"/prompts.json"
with open(prompt_path, "r") as file:
    data = json.load(file)

class Prompts:
    def get_description(self, label):
        return data["few_shot_prompting"]["labels"][label]["description"]

    def get_output_structure(self, label, output_type="few_shot_prompting"):
        return data[output_type]["labels"][label]["output_structure"]

    def content(self, description, output_structure):
        output_structure = json.dumps(output_structure)
        description_prompt = "Please extract " + description.lower() + " Answer must follow the following format:\n[" + output_structure + "...]"
        return description_prompt

    def get_system_prompt(self, type):
        return data[type]["system_prompt"]

    def prompt_only(self, label, medical_text):
        """Returns the messages for prompt only
        label: string
        medical_text: string
        """
        message = []
        description = self.get_description(label)
        output_structure = self.get_output_structure(label)
        message.append({"role": "system", "content": self.get_system_prompt("prompt_only")})
        message.append({"role": "user", "content": self.content(description, output_structure) + "\n\nMedical text:\n" + medical_text})
        return message

    def get_example(self, label, example_number):
        return data["few_shot_prompting"]["labels"][label]["example_" + str(example_number)]

    def few_shot_prompting(self, label, medical_text):
        """Returns the messages for few shot prompting with 3 examples.
        label: string
        medical_text: string
        """
        message = []
        description = self.get_description(label)
        output_structure = self.get_output_structure(label)
        message.append({"role": "system", "content": self.get_system_prompt("prompt_only")})
        message.append({"role": "user", "content": self.content(description, output_structure) + "\n\nMedical text:\n" + self.get_example(label, 1)["input"]})
        message.append({"role": "assistant", "content": self.get_example(label, 1)["output"]})
        message.append({"role": "user", "content": self.content(description, output_structure) + "\n\nMedical text:\n" + self.get_example(label, 4)["input"]})
        message.append({"role": "assistant", "content": self.get_example(label, 4)["output"]})
        message.append({"role": "user", "content": self.content(description, output_structure) + "\n\nMedical text:\n" + self.get_example(label, 2)["input"]})
        message.append({"role": "assistant", "content": self.get_example(label, 2)["output"]})
        message.append({"role": "user", "content": self.content(description, output_structure) + "\n\nMedical text:\n" + medical_text})
        return message
    
    def conversational_training_prompt(self, labels, output_structure, medical_text):
        output_structure = json.dumps(output_structure)
        labels_string = ', '.join(str(label) for label in labels)
        instruction_content = "Please extract the following entities: " + labels_string + ". Answer must follow the following format:\n[" + output_structure + "...]"
        prompt = instruction_content + "\n\nMedical text:\n" + medical_text
        return prompt
    
    def conversational_training(self, labels, medical_text, system_output):
        # Function is for training the model on the conversational prompt
        message = []
        output_structure = self.get_output_structure("no_label", output_type="conversational_training")
        message.append({"role": "system", "content": self.get_system_prompt("conversational_training")})
        message.append({"role": "user", "content": self.conversational_training_prompt(labels, output_structure, medical_text)})
        message.append({"role": "assistant", "content": json.dumps(system_output)})
        return message
    
    def conversational_prompt(self, labels, medical_text):
        # Function is for evaluating the model on the conversational prompt
        message = []
        output_structure = self.get_output_structure("no_label", output_type="conversational_training")
        message.append({"role": "system", "content": self.get_system_prompt("conversational_training")})
        message.append({"role": "user", "content": self.conversational_training_prompt(labels, output_structure, medical_text)})
        return message
    
    def instruction_training_prompt(self, labels, output_structure, medical_text):
        output_structure = json.dumps(output_structure)
        labels_string = ', '.join(str(label) for label in labels)
        instruction_content = "Please extract the following entities: " + labels_string + ". Answer must follow the following format:\n[" + output_structure + "...]"
        prompt = self.get_system_prompt("instruction_training") + "\n\n" + instruction_content + "\n\nMedical text:\n" + medical_text
        return prompt
    
    def instruction_training(self, labels, medical_text, system_output):
        # Function is for training the model on the instruction prompt
        message = {}
        output_structure = self.get_output_structure("no_label", output_type="instruction_training")
        message["prompt"] = self.instruction_training_prompt(labels, output_structure, medical_text)
        message["completion"] = json.dumps(system_output)
        return message
    
    def instruction_prompt(self, labels, medical_text):
        # Function is for evaluating the model on the instruction prompt
        message = {}
        output_structure = self.get_output_structure("no_label", output_type="instruction_training")
        message["prompt"] = self.instruction_training_prompt(labels, output_structure, medical_text)
        return message
    
    