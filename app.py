from lcserve import serving
import os

from langchain import OpenAI
from chain.SqlChain import SqlChain

import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

import logging
logging.getLogger().setLevel(logging.DEBUG)


@serving
def completion_messages(inputs: dict, query: str) -> dict:
    llm = OpenAI(
        verbose=True,
        temperature=0,
        engine="gpt-35-turbo-16k",
        model_name="gpt-3.5-turbo-16k",
        headers={
            "Helicone-Auth": "Bearer " + os.environ.get("HELICONE_API_KEY"),
            "Helicone-OpenAI-Api-Base": "https://neucloud.openai.azure.com",
            "Helicone-Cache-Enabled": "true"
        }
    )
    db_chain = SqlChain(llm=llm, ds_id=inputs['ds_id'])
    response = db_chain(query)
    return response


if __name__ == '__main__':
    pass
