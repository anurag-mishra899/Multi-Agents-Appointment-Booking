from pydantic_settings import BaseSettings
from functools import lru_cache
import yaml 

with open('devConfig.yaml') as f:
    configParser = yaml.load(f, Loader=yaml.SafeLoader)

class Settings(BaseSettings):
    AZURE_OPENAI_API_KEY: str =  configParser['AZURE_OPENAI_API_KEY'] 
    AZURE_OPENAI_ENDPOINT: str =  configParser['AZURE_OPENAI_ENDPOINT'] 
    AZURE_OPENAI_VERSION: str =  configParser['AZURE_OPENAI_VERSION'] 
    AZURE_GPT4O_MODEL: str =  configParser['AZURE_GPT4O_MODEL'] 
    AZURE_OPENAI_EMBEDDINGS_MODEL: str =  configParser['AZURE_OPENAI_EMBEDDINGS_MODEL'] 
    LANGCHAIN_ENDPOINT: str =  configParser['LANGCHAIN_ENDPOINT'] 
    LANGCHAIN_API_KEY: str =  configParser['LANGCHAIN_API_KEY'] 
    LANGCHAIN_PROJECT: str =  configParser['LANGCHAIN_PROJECT'] 
    AI_SEARCH_KEY: str =  configParser['AI_SEARCH_KEY'] 
    AI_SEARCH_ENDPOINT: str =  configParser['AI_SEARCH_ENDPOINT'] 


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()



