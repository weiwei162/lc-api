from typing import List, Optional

from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from langchain.schema.language_model import BaseLanguageModel

from sqlalchemy import create_engine, text
import pandas as pd
from core.lc.nc import get_db_uri


PREFIX_FUNCTIONS = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
Only use the functions you have been provided with.
Don't make assumptions about what values to use with functions. Ask for clarification if a user request is ambiguous.
"""


class PandasAgent():
    def from_llm(
        llm: BaseLanguageModel,
        ds_id: str,
        include_tables: Optional[List[str]] = None,
        k: int = 5,
    ) -> any:
        uri = get_db_uri(ds_id)

        engine = create_engine(uri)
        with engine.connect() as conn:
            dfs = [pd.read_sql_query(text(f"SELECT * FROM {table_name}"), conn)
                   for table_name in include_tables]

        agent_executor = create_pandas_dataframe_agent(
            llm,
            dfs[0],
            prefix=PREFIX_FUNCTIONS,
            max_iterations=5,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )
        agent_executor.return_intermediate_steps = True
        return agent_executor
