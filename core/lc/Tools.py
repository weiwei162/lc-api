import pydantic
from typing import Any, Type

from langchain.tools import BaseTool

from core.lc.schema import PlotConfig
from core.lc.render import draw_plotly

from sqlalchemy import create_engine, text
import pandas as pd


class PlotTool(BaseTool):
    name = "generate_chart"
    description = """
    Generate the chart with given parameters
    """
    args_schema: Type[pydantic.BaseModel] = PlotConfig

    return_direct = True

    uri: str

    def _run(self, **kwargs):
        sql = kwargs["sql"]
        engine = create_engine(self.uri)
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql), conn)

        config = PlotConfig.parse_obj(kwargs)
        figure = draw_plotly(df, config, False)
        return {"sql": sql, "result": "![image]({})".format(figure)}

    def _arun(self):
        raise NotImplementedError("does not support async")
