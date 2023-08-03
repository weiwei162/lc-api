import bentoml
import os

from langchain import OpenAI
from core.lc.SqlChain import SqlChain

import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
langchain.verbose = True
langchain.debug = True

# import logging
# logging.getLogger().setLevel(logging.DEBUG)

svc = bentoml.Service("lc")


@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def completion_messages(inputs: dict) -> dict:
    llm = OpenAI(
        verbose=True,
        temperature=0,
        engine="gpt-35-turbo-16k",
        model_name="gpt-3.5-turbo-16k",
        # headers={
        #     "Helicone-Auth": "Bearer " + os.environ.get("HELICONE_API_KEY"),
        #     "Helicone-OpenAI-Api-Base": "https://neucloud.openai.azure.com",
        #     "Helicone-Cache-Enabled": "true"
        # }
    )
    db_chain = SqlChain.from_llm(llm, inputs['inputs']['ds_id'])
    response = db_chain.invoke({"question": inputs['query']})
    return response


if __name__ == '__main__':
    pass
