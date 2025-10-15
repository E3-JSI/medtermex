import os

from dotenv import find_dotenv, load_dotenv


# TODO: Might need to remove this (if it is not used)
def load_env_file():
    """
    Loads environment variables from a specified .env file.

    The function attempts to load environment variables from a file specified by the
    'ENV_FILE' environment variable. If 'ENV_FILE' is not set, it will use the first
    .env file found in the current directory or its parents. If the file exists, the
    environment variables are loaded and a success message is printed. If the file
    does not exist, an error message is printed.

    Returns:
        None
    """
    env_file = os.getenv("ENV_FILE", find_dotenv())
    if env_file and os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"Loaded environment variables from {env_file}")
    else:
        print(f"Environment file {env_file} not found")


load_env_file()
