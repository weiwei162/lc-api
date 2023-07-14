# app.py
import os
from sqlalchemy import create_engine, text
import json
import base64
import hashlib
from Crypto.Cipher import AES
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.chains import SQLDatabaseSequentialChain

import logging
logging.getLogger().setLevel(logging.DEBUG)

from lcserve import serving


@serving
def data_query(message: str, id: str) -> str:
    engine = create_engine(
        os.environ.get("NC_DB"), echo=True
    )
    with engine.connect() as conn:
        result = conn.execute(
            text('SELECT * FROM nc_bases_v2 where id = :id'), {"id": id}
        ).first()

    config = json.loads(decrypt_aes_ciphertext(result.config))['connection']
    dbType = {
        'oracle': 'oracle+cx_oracle',
        'pg': 'postgresql',
        'mysql2': 'mysql+pymysql'
    }.get(result.type)

    uri = '{dbType}://{user}:{pw}@{host}:{port}/{db}'.format(
        dbType=dbType,
        user=config['user'],
        pw=config['password'],
        host=config['host'],
        port=config['port'],
        db=config['database']
    )
    db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=0)
    llm = OpenAI(
        verbose=True,
        temperature=0,
        engine="gpt-35-turbo-16k",
        model_name="gpt-3.5-turbo-16k",
        headers={
            "Helicone-Auth": "Bearer " + os.environ.get("HELICONE_API_KEY"),
            "Helicone-OpenAI-Api-Base": "https://neucloud.openai.azure.com"
        }
    )
    db_chain = SQLDatabaseSequentialChain.from_llm(
        llm,
        db,
        verbose=True,
        return_direct=True,
        top_k=10
    )
    response = db_chain.run(message)

    return response


def decrypt_aes_ciphertext(ciphertext_base64):
    # NC_AUTH_JWT_SECRET
    password = os.environ.get("NC_AUTH_JWT_SECRET").encode('ASCII')
    ciphertext = base64.b64decode(ciphertext_base64)
    ciphertext_hex = ciphertext.hex()
    salt = bytes.fromhex(ciphertext_hex[16:32])

    derived_key = b''
    data = b''

    while len(derived_key) < 48:
        md5 = hashlib.md5()
        md5.update(data + password + salt)
        data = md5.digest()
        derived_key += data

    key = derived_key[:32]
    iv = derived_key[32:]

    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(bytes.fromhex(ciphertext_hex[32:])).decode('ASCII')
    return plaintext[:-ord(plaintext[-1])]


if __name__ == '__main__':
    pass