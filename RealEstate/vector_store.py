# vector_store.py
# ------------------------------------------------------------------
# Module quản lý Pinecone + LangChain embeddings cho bất động sản
# ------------------------------------------------------------------

import uuid
from typing import Any, List, Optional

import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings


class RealEstateVectorStore:
    """
    Bao gói:
    - HuggingFaceEmbeddings (LangChain)
    - Pinecone index
    - Hàm rebuild index từ DataFrame
    - Hàm search (vector search + filter theo intent)
    """

    def __init__(
        self,
        api_key: str,
        index_name: str = "real-estate-rag",
        cloud: str = "aws",
        region: str = "us-east-1",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        if not api_key:
            raise ValueError("Pinecone API key is required")

        # 1) Embedding model (LangChain)
        self.embedder = HuggingFaceEmbeddings(model_name=embed_model_name)

        # 2) Init Pinecone client
        self.pc = Pinecone(api_key=api_key)

        # 3) Tạo index nếu chưa có
        test_vec = self.embedder.embed_query("ping")
        dim = len(test_vec)

        existing = [i["name"] for i in self.pc.list_indexes()]
        if index_name not in existing:
            self.pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self.index = self.pc.Index(index_name)
        self.index_name = index_name

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def rebuild(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        """
        Xây lại index từ DataFrame.
        Trả về số lượng listing được index.
        """
        # Xoá toàn bộ vector cũ
        try:
            self.index.delete(delete_all=True)
        except Exception:
            pass

        vectors = []
        for i, row in df.iterrows():
            text = str(row["doc_text"])
            vec = self.embedder.embed_query(text)
            meta = {
                "row_index": int(i),
                "Location": str(row.get("Location", "")),
                "Price": str(row.get("Price", "")),
                "Type": str(row.get("Type of Hou", row.get("Type of House", ""))),
                "Bedrooms": row.get("bedrooms"),
                "Toilets": row.get("toilets"),
                "Floors": row.get("floors"),
            }
            vectors.append((str(uuid.uuid4()), vec, meta))

        for start in range(0, len(vectors), batch_size):
            end = start + batch_size
            batch = vectors[start:end]
            self.index.upsert(vectors=batch)

        return len(vectors)

    def is_empty(self) -> bool:
        """Kiểm tra index có rỗng không."""
        try:
            stats = self.index.describe_index_stats()
            return stats.get("total_vector_count", 0) == 0
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Search + filter theo intent
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        df: pd.DataFrame,
        intent: Optional[Any] = None,
        k: int = 20,
    ) -> pd.DataFrame:
        """
        - query: câu hỏi người dùng
        - df: DataFrame gốc (để map row_index -> row)
        - intent: object có thể có các attribute:
          max_price_billion, min_bedrooms, min_toilets, must_frontage
        - k: top_k vector search
        """
        # 1) Vector search
        try:
            vec = self.embedder.embed_query(query)
            res = self.index.query(
                vector=vec,
                top_k=k,
                include_metadata=True,
            )
            matches = res.get("matches", [])
        except Exception:
            matches = []

        rows: List[int] = []
        for m in matches:
            meta = m.get("metadata") or {}
            ri = meta.get("row_index")
            if ri is not None:
                rows.append(int(ri))

        cand = df.loc[rows].copy() if rows else df.copy()

        # 2) Filter theo intent (nếu có)
        if intent is not None:
            max_price_billion = getattr(intent, "max_price_billion", None)
            min_bedrooms = getattr(intent, "min_bedrooms", None)
            min_toilets = getattr(intent, "min_toilets", None)
            must_frontage = getattr(intent, "must_frontage", False)

            if max_price_billion is not None and "price_billion" in cand.columns:
                cand = cand[cand["price_billion"] <= float(max_price_billion)]
            if min_bedrooms is not None and "bedrooms" in cand.columns:
                cand = cand[cand["bedrooms"] >= int(min_bedrooms)]
            if min_toilets is not None and "toilets" in cand.columns:
                cand = cand[cand["toilets"] >= int(min_toilets)]
            if must_frontage:
                type_col = "Type of Hou" if "Type of Hou" in cand.columns else (
                    "Type of House" if "Type of House" in cand.columns else None
                )
                if type_col:
                    cand = cand[cand[type_col].str.contains("mặt tiền", case=False, na=False)]

        return cand
