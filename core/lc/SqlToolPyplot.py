from typing import List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.schema.runnable import RunnableMap
from operator import itemgetter
from langchain.tools import tool
from langchain.agents import AgentType, initialize_agent

from langchain.schema.language_model import BaseLanguageModel
from langchain import SQLDatabase

from sqlalchemy import create_engine, text
import pandas as pd
from core.lc.nc import get_db_uri

import base64
import io
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


_DECIDER_TEMPLATE = """Given the below input question and list of potential tables, output a comma separated list of the table names that may be necessary to answer this question.
Question: {question}
Table Names: {table_names}
Relevant Table Names:"""

DECIDER_PROMPT = ChatPromptTemplate.from_template(_DECIDER_TEMPLATE)


class SqlToolPyplot():
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

        @tool(return_direct=True)
        def line_chart_executor(title: str, sql: str):
            """According to the user's analysis goal, generate a corresponding line chart to analyze the SQL.

            Args:
                title: Data analysis title name
                sql: Data analysis sql
            """
            engine = create_engine(uri)
            with engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn)
                columns = df.columns.tolist()
                if df.size <= 0:
                    raise ValueError("No Data！" + sql)

            sns.set(style="ticks", color_codes=True)
            plt.subplots(figsize=(8, 5), dpi=100)
            sns.lineplot(df, x=columns[0], y=columns[1])
            plt.title(title)

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            data = base64.b64encode(buf.getvalue()).decode("ascii")

            html = f"""<img style='max-width: 120%; max-height: 80%;' src="data:image/png;base64,{data}" />"""

            return {"sql": sql, "result": html}

        @tool(return_direct=True)
        def histogram_executor(title: str, sql: str):
            """According to the user's analysis goal, generate the corresponding histogram to analyze the SQL.

            Args:
                title: Data analysis title name
                sql: Data analysis sql
            """
            engine = create_engine(uri)
            with engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn)
                columns = df.columns.tolist()
                if df.size <= 0:
                    raise ValueError("No Data！" + sql)

            sns.set(context="notebook", style="ticks", color_codes=True)
            plt.subplots(figsize=(8, 5), dpi=100)
            sns.barplot(df, x=columns[0], y=columns[1])
            plt.title(title)

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            data = base64.b64encode(buf.getvalue()).decode("ascii")

            html = f"""<img style='max-width: 120%; max-height: 80%;' src="data:image/png;base64,{data}" />"""

            return {"sql": sql, "result": html}

        tools = [line_chart_executor, histogram_executor]
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
