import bentoml
import os

from langchain.chat_models import ChatOpenAI
from core.lc.SqlChain import SqlChain
from core.lc.SqlAgent import SqlAgent
from core.lc.SqlToolPyplot import SqlToolPyplot
from core.lc.SqlTool import SqlTool
from core.lc.PandasAgent import PandasAgent

import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
langchain.verbose = True
langchain.debug = True

# import logging
# logging.getLogger().setLevel(logging.DEBUG)

svc = bentoml.Service("lc")


@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def completion_messages(req: dict) -> dict:
    llm = ChatOpenAI(
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
    inputs = req['inputs']
    db_chain = PandasAgent.from_llm(
        llm, inputs['ds_id'], inputs['include_tables'].split(','))
    response = db_chain.invoke({"input": req['query']})
    return response


if __name__ == '__main__':
    pass
