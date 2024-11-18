# noqa: E501
START_PROMPT = "### Your goal is to extract structured information from the user's input that matches json format. When extracting information please make sure it matches the type information exactly. Extract the following entities: "


class Prompts:
    def age(input):
        age = (
            START_PROMPT
            + f"""Age. Do not add any extra entities.

            ### Input:\n{input}

            ### Output:"""
        )
        return age

    def sex(input):
        sex = (
            START_PROMPT
            + f"""Sex. Do not add any extra entities.

            ### Input:\n{input}

            ### Output:"""
        )
        return sex

    def biological_structure(input):
        bio = (
            START_PROMPT
            + f"""Biological_structure. Do not add any extra entities.

            ### Input:\n{input}

            ### Output:"""
        )
        return bio

    def sign_symptom(input):
        sign = (
            START_PROMPT
            + f"""Sign_symptom. Do not add any extra entities.

            ### Input:\n{input}

            ### Output:"""
        )
        return sign

    def diagnostic_procedure(input):
        diagn = (
            START_PROMPT
            + f"""Diagnostic_procedure. Do not add any extra entities.

            ### Input:\n{input}

            ### Output:"""
        )
        return diagn

    def lab_value(input):
        lab = (
            START_PROMPT
            + f"""Lab_value. Do not add any extra entities.

            ### Input:\n{input}

            ### Output:"""
        )
        return lab
