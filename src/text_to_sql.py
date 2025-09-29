# vanna_pgvector_store_model.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text, bindparam, String, DateTime
from sqlalchemy.orm import registry, mapped_column, Mapped, Session
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer

from vanna.base import VannaBase
from groq import Groq



def _uuid():
    import uuid
    return str(uuid.uuid4())


class Postgre_VectorStore(VannaBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config=config)
        if "database_url" not in config:
            raise ValueError("config must include 'database_url'")
        
        self.engine = create_engine(config["database_url"], echo=False, future=True)
        self.prefix = config.get("table_prefix", "vanna")
        self.use_hnsw = bool(config.get("use_hnsw", False))
        self.lists = int(config.get("lists", 100))
        self.hnsw_m = int(config.get("hnsw_m", 16))
        self.hnsw_efc = int(config.get("hnsw_ef_construction", 64))
        self.embedding_model = SentenceTransformer(config.get("embedding_model", "all-MiniLM-L6-v2"))

        if hasattr(self.embedding_model, "get_sentence_embedding_dimension"):
            self._dim = self.embedding_model.get_sentence_embedding_dimension()
        else:
            tmp = self.embedding_model.encode(["_"], normalize_embeddings=True, convert_to_numpy=True)
            self._dim = int(tmp.shape[1])

        self._mapper_registry: Optional[registry] = None
        self._DDLRow: Optional[Type] = None
        self._DocRow: Optional[Type] = None
        self._QSQLRow: Optional[Type] = None

        self._bind_schema(self._dim)

        print(f"VectorStore bound to {config['database_url']}")

    # ---------- Embeddings ----------
    def generate_embedding(self, data: Any) -> np.ndarray:
        if isinstance(data, str):
            data = [data]
        embs = self.embedding_model.encode(
            data,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if embs.shape[1] != self._dim:
            raise ValueError(f"Embedding dim {embs.shape[1]} != bound dim {self._dim}")
        return embs

    # ---------- Public API ----------
    def add_ddl(self, ddl: str) -> str:
        emb = self.generate_embedding(ddl)[0].tolist()
        with Session(self.engine) as ses, ses.begin():
            row = self._DDLRow(ddl=ddl, embedding=emb)
            ses.add(row)
            return row.id

    def add_documentation(self, doc: str) -> str:
        emb = self.generate_embedding(doc)[0].tolist()
        with Session(self.engine) as ses, ses.begin():
            row = self._DocRow(doc=doc, embedding=emb)
            ses.add(row)
            return row.id

    def add_question_sql(self, question: str, sql: str) -> str:
        emb = self.generate_embedding(question)[0].tolist()
        with Session(self.engine) as ses, ses.begin():
            row = self._QSQLRow(question=question, sql=sql, embedding=emb)
            ses.add(row)
            return row.id

    def get_related_ddl(self, question: str, k: int = 5) -> list[str]:
        """
        Return the most similar DDL strings (column: ddl) from table: <prefix>_ddl_<dim>.
        """
        qvec = self.generate_embedding(question)[0].tolist()
        stmt = (
            text(f"""
                SELECT ddl
                FROM {self._t('ddl')}
                ORDER BY embedding <=> :q
                LIMIT :k
            """)
            .bindparams(
                bindparam("q", value=qvec, type_=Vector(self._dim)),
                bindparam("k", value=k),
            )
        )
        with Session(self.engine) as ses:
            # use .mappings() so we can access by column name reliably
            rows = ses.execute(stmt).mappings().all()
            return [r["ddl"] for r in rows]


    def get_related_documentation(self, question: str, k: int = 5) -> list[str]:
        """
        Return the most similar documentation strings (column: doc) from table: <prefix>_docs_<dim>.
        """
        qvec = self.generate_embedding(question)[0].tolist()
        stmt = (
            text(f"""
                SELECT doc
                FROM {self._t('docs')}
                ORDER BY embedding <=> :q
                LIMIT :k
            """)
            .bindparams(
                bindparam("q", value=qvec, type_=Vector(self._dim)),
                bindparam("k", value=k),
            )
        )
        with Session(self.engine) as ses:
            rows = ses.execute(stmt).mappings().all()
            return [r["doc"] for r in rows]


    def get_similar_question_sql(self, question: str, k: int = 5) -> list[dict[str, str]]:
        """
        Return similar (question, sql) pairs from table: <prefix>_qsql_<dim>.
        Column names match ORM: question, sql.
        """
        qvec = self.generate_embedding(question)[0].tolist()
        stmt = (
            text(f"""
                SELECT question, sql
                FROM {self._t('qsql')}
                ORDER BY embedding <=> :q
                LIMIT :k
            """)
            .bindparams(
                bindparam("q", value=qvec, type_=Vector(self._dim)),
                bindparam("k", value=k),
            )
        )
        with Session(self.engine) as ses:
            rows = ses.execute(stmt).mappings().all()
            return [{"question": r["question"], "sql": r["sql"]} for r in rows]


    def get_training_data(self) -> pd.DataFrame:
        """
        Matches ORM: columns id, question, sql, created_at from <prefix>_qsql_<dim>.
        """
        with Session(self.engine) as ses:
            rows = ses.execute(
                text(f"SELECT id, question, sql, created_at FROM {self._t('qsql')}")
            ).mappings().all()
        return pd.DataFrame(rows)

    def remove_training_data(self, id: str) -> bool:
        with Session(self.engine) as ses, ses.begin():
            n = ses.execute(text(f"DELETE FROM {self._t('qsql')} WHERE id = :id"), {"id": id}).rowcount
            return n > 0

    # ---------- Internals ----------
    def _t(self, kind: str) -> str:
        return f"{self.prefix}_{kind}_{self._dim}"

    def _bind_schema(self, dim: int):
        APP_SCHEMA = "public"
        reg = registry()

        @reg.mapped
        class DDLRow:
            __tablename__ = f"{self.prefix}_ddl_{dim}"
            __table_args__ = {"schema": APP_SCHEMA}   # explicit: always in public
            id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
            ddl: Mapped[str] = mapped_column(String)
            embedding: Mapped[List[float]] = mapped_column(Vector(dim))
            created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

        @reg.mapped
        class DocRow:
            __tablename__ = f"{self.prefix}_docs_{dim}"
            __table_args__ = {"schema": APP_SCHEMA}
            id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
            doc: Mapped[str] = mapped_column(String)
            embedding: Mapped[List[float]] = mapped_column(Vector(dim))
            created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

        @reg.mapped
        class QSQLRow:
            __tablename__ = f"{self.prefix}_qsql_{dim}"
            __table_args__ = {"schema": APP_SCHEMA}
            id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
            question: Mapped[str] = mapped_column(String)
            sql: Mapped[str] = mapped_column(String)
            embedding: Mapped[List[float]] = mapped_column(Vector(dim))
            created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

        self._mapper_registry = reg
        self._DDLRow, self._DocRow, self._QSQLRow = DDLRow, DocRow, QSQLRow

        # ----- 1) Ensure pgvector is installed in *public* (autocommit) -----
        with self.engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;"))

        # Precompute fully-qualified table names for raw SQL (no quotes needed if all lowercase/underscores)
        tddl  = f"{APP_SCHEMA}.{self._t('ddl')}"
        tdocs = f"{APP_SCHEMA}.{self._t('docs')}"
        tqsql = f"{APP_SCHEMA}.{self._t('qsql')}"

        # ----- 2) Create tables + indexes on the SAME connection/transaction -----
        with self.engine.begin() as conn:
            # (Optional, defensive) make sure public is on search_path for this tx
            # conn.execute(text("SET search_path TO public, pg_catalog"))

            reg.metadata.create_all(bind=conn)

            if self.use_hnsw:
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {self.prefix}_ddl_{dim}_hnsw_cos
                    ON {tddl} USING hnsw (embedding public.vector_cosine_ops)
                    WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_efc});
                """))
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {self.prefix}_docs_{dim}_hnsw_cos
                    ON {tdocs} USING hnsw (embedding public.vector_cosine_ops)
                    WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_efc});
                """))
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {self.prefix}_qsql_{dim}_hnsw_cos
                    ON {tqsql} USING hnsw (embedding public.vector_cosine_ops)
                    WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_efc});
                """))
            else:
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {self.prefix}_ddl_{dim}_ivf_cos
                    ON {tddl} USING ivfflat (embedding public.vector_cosine_ops)
                    WITH (lists = {self.lists});
                """))
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {self.prefix}_docs_{dim}_ivf_cos
                    ON {tdocs} USING ivfflat (embedding public.vector_cosine_ops)
                    WITH (lists = {self.lists});
                """))
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {self.prefix}_qsql_{dim}_ivf_cos
                    ON {tqsql} USING ivfflat (embedding public.vector_cosine_ops)
                    WITH (lists = {self.lists});
                """))

                # Analyze after index creation (helps IVF/HNSW planning)
                conn.execute(text(f"ANALYZE {tddl};"))
                conn.execute(text(f"ANALYZE {tdocs};"))
                conn.execute(text(f"ANALYZE {tqsql};"))


class Groq_Chat(VannaBase):
    def __init__(self, config):
        
        if "api_key" not in config:
            raise ValueError("config must include 'api_key'")
        
        if "model" not in config:
            raise ValueError("config must include 'model'")
        
        if "temperature" not in config:
            config["temperature"] = 0
        

        self.client = Groq(api_key=config["api_key"])
        self.config = config
        
        VannaBase.__init__(self, config=config)

    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}
    
    def submit_prompt(self, prompt, **kwargs):
        
        if prompt is None:
            raise Exception("Prompt is None")

        if len(prompt) == 0:
            raise Exception("Prompt is empty")

        # Count the number of tokens in the message log
        # Use 4 as an approximation for the number of characters per token
        # num_tokens = 0
        # for message in prompt:
        #     num_tokens += len(message["content"]) / 4

        # print(f"Using model {self.config['model']} for {num_tokens} tokens (approx)")
        
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=prompt,
            stop=None,
            temperature=self.config["temperature"],
        )

        return response.choices[0].message.content


# if __name__ == "__main__":
    
#     from settings import settings
#     config = {
#         "api_key": settings.groq_api_key,
#         "model": "llama-3.1-8b-instant",
#         "database_url": settings.postgres_url,
#         "embedding_model": "Qwen/Qwen3-Embedding-0.6B"
#     }

    # vn = VannaGroq(config=config)
    # vn.train(documentation="i wnt to test my vanna ")
    # print(vn.get_related_documentation("i wnt to test my vanna"))