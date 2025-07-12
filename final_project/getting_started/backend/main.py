from fastapi.responses import HTMLResponse
from config import INDEX_NAME_DEFAULT, INDEX_NAME_N_GRAM, INDEX_NAME_EMBEDDING
import torch
from sentence_transformers import SentenceTransformer
from utils import get_es_client

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

@app.get("/api/v1/regular_search")
async def regular_search(search_query: str, skip: int = 0, limit: int = 10, year: str | None = None) -> dict:
    es = get_es_client(max_retries=2, sleep_time=1)
    query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": search_query,
                            "fields": ["title", "explanation"]
                        }
                    }
                ]
            }   
    }   
    if year:
        query["bool"]["filter"] = [
            {
                "range": {
                    "date": {
                        "gte": f"{year}-01-01",
                        "lte": f"{year}-12-31",
                        "format": "yyyy-MM-dd"
                    }
                }
            }
        ]
    response = es.search(
        index=INDEX_NAME_N_GRAM,
        body={
            "query": query,
            "from": skip,
            "size": limit,
        },
        filter_path=[
            "hits.hits._source",
            "hits.hits._score",
            "hits.total",
        ],
    )
    total_hits = get_total_hits(response)
    max_pages = calculate_max_pages(total_hits, limit)
    return {
        "hits": response["hits"].get("hits", []),
        "max_pages": max_pages,
    }

@app.get("/api/v1/get_docs_per_year_count/")
async def get_docs_per_year_count(search_query: str) -> dict:
    try:
        es = get_es_client(max_retries=2, sleep_time=1)
        query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": search_query,
                            "fields": ["title", "explanation"]
                        }
                    }
                ]
            }
        }

        response = es.search(
        index=INDEX_NAME_N_GRAM,
        body = {
            "query": query,
            "aggs": {
                "docs_per_year": {
                    "date_histogram": {
                        "field": "date",
                        "calendar_interval": "year",
                        "format": "yyyy"
                    }
                }
            }
            
        },
        filter_path=[
            "aggregations.docs_per_year"
        ]
    )
        return {"docs_per_year": extract_docs_per_year(response)}
    except Exception as e:
        return HTMLResponse(
            content=f"Error: {str(e)}",
            status_code=500
        )
    

def extract_docs_per_year(response: dict) -> dict:
    aggregations = response.get("aggregations", {})
    docs_per_year = aggregations.get("docs_per_year", {})
    buckets = docs_per_year.get("buckets", [])
    return {bucket["key_as_string"]: bucket["doc_count"] for bucket in buckets}

def get_total_hits(response: dict) -> int:
    return response["hits"]["total"]["value"]


def calculate_max_pages(total_hits: int, limit: int) -> int:
    return (total_hits + limit - 1) // limit


@app.get("/api/v1/semantic_search/")
async def semantic_search(search_query: str, skip: int = 0, limit: int = 10, year: str | None = None) -> dict:
    es = get_es_client(max_retries=2, sleep_time=1)
    embedded_query = model.encode(search_query)
    query = {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "query_vector": embedded_query,
                            "field": "embedding",
                            "k": 1e4,
                        }
                    }
                ]
            }   
    }   
    if year:
        query["bool"]["filter"] = [
            {
                "range": {
                    "date": {
                        "gte": f"{year}-01-01",
                        "lte": f"{year}-12-31",
                        "format": "yyyy-MM-dd"
                    }
                }
            }
        ]
    response = es.search(
        index=INDEX_NAME_EMBEDDING,
        body={
            "query": query,
            "from": skip,
            "size": limit,
        },
        filter_path=[
            "hits.hits._source",
            "hits.hits._score",
            "hits.total",
        ],
    )
    total_hits = get_total_hits(response)
    max_pages = calculate_max_pages(total_hits, limit)
    return {
        "hits": response["hits"].get("hits", []),
        "max_pages": max_pages,
    }