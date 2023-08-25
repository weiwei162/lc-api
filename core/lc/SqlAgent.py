from typing import List, Optional

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

from langchain.schema.language_model import BaseLanguageModel
from langchain import SQLDatabase

from core.lc.nc import get_db_uri


SqlPrefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Always limit your query to at most {top_k} results using the SELECT TOP in SQL Server syntax.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
If you get a "no such table" error, rewrite your query by using the table in quotes.
DO NOT use a column name that does not exist in the table.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite a different query and try again.
Observations from the database should be in the form of a JSON with following keys: "column_name", "column_value"
DO NOT try to execute the query more than three times.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
If the question does not seem related to the database, just return "I don't know" as the answer.
If you cannot find a way to answer the question, just return the best answer you can find after trying at least three times."""


class SqlAgent():
    def from_llm(
        llm: BaseLanguageModel,
        ds_id: str,
        include_tables: Optional[List[str]] = None,
        k: int = 5,
    ) -> any:
        uri = get_db_uri(ds_id)

        db = SQLDatabase.from_uri(
            uri,
            # {'echo': True},
            include_tables=include_tables,
            # sample_rows_in_table_info=0
            view_support=True,
        )

        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            prefix=SqlPrefix,
            top_k=k,
            max_iterations=5,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )
        agent_executor.return_intermediate_steps = True
        return agent_executor
