from typing import List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.schema.runnable import RunnableMap
from operator import itemgetter
from langchain.agents import AgentType, initialize_agent

from langchain.schema.language_model import BaseLanguageModel
from langchain import SQLDatabase

from core.lc.nc import get_db_uri
from core.lc.Tools import PlotTool


_DECIDER_TEMPLATE = """Given the below input question and list of potential tables, output a comma separated list of the table names that may be necessary to answer this question.
Question: {question}
Table Names: {table_names}
Relevant Table Names:"""

DECIDER_PROMPT = ChatPromptTemplate.from_template(_DECIDER_TEMPLATE)


class SqlTool():
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

        prompt_msgs = """please follow the constraints of the system's input for your answers.
        Constraints:
        1.Combine user question and improved information to generate {dialect} SQL statements for data analysis.
        2.Use as few tables as possible when querying.
        3.Try to use left joins but include the complete range of the first query target.
        4.Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
        5.In SQL, it is necessary to prefix the table name before the column name, for example, SELECT table_alias.column_name from table_name AS table_alias.
        6.Only use the following tables:
        {table_info}

        Only use the functions you have been provided with.
        Ensure the response is correct json and can be parsed by Python json.loads"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_msgs),
        ])

        tools = [PlotTool(uri=uri)]
        agent = initialize_agent(tools, llm, AgentType.OPENAI_FUNCTIONS, agent_kwargs={
                                 "system_message": prompt})

        _table_names = db.get_usable_table_names()
        table_names = ", ".join(_table_names)

        table_names_from_chain = {"question": itemgetter(
            "input"), "table_names": lambda _: table_names} | DECIDER_PROMPT | llm | CommaSeparatedListOutputParser()

        return (
            RunnableMap({
                "input": itemgetter("input"),
                "table_names_to_use": table_names_from_chain,
            })
            | {
                "input": itemgetter("input"),
                "table_info": lambda x: db.get_table_info(
                    table_names=x.get("table_names_to_use")
                ),
                "dialect": lambda _: db.dialect
            }
            | agent
            | RunnableMap({
                "data": itemgetter("output"),
            })
        )
