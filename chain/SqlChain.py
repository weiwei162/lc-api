from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.schema.language_model import BaseLanguageModel
from langchain import SQLDatabase
from langchain.chains import SQLDatabaseSequentialChain
from langchain.callbacks.manager import CallbackManagerForChainRun

from sqlalchemy import create_engine
import pandas as pd
from chain.nc import get_db_uri


class SqlChain(Chain):
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    llm: Optional[BaseLanguageModel] = None
    ds_id: str = None
    uri: str = None

    @property
    def _chain_type(self) -> str:
        return "SqlChain"

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        uri = get_db_uri(self.ds_id)

        db = SQLDatabase.from_uri(
            uri,
            {'echo': True},
            # sample_rows_in_table_info=0
        )
        db_chain = SQLDatabaseSequentialChain.from_llm(
            self.llm,
            db,
            verbose=True,
            return_intermediate_steps=True,
            top_k=10
        )
        response = db_chain(inputs[self.input_key], callbacks=_run_manager.get_child(),
                            return_only_outputs=True)

        sql = response['intermediate_steps'][1]
        engine = create_engine(uri, echo=True)
        with engine.connect() as conn:
            df = pd.read_sql_query(sql, conn)
            response['data'] = df.to_dict('records')

        return response
