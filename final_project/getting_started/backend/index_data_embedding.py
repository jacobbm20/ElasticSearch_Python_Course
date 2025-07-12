from pprint import pprint
from typing import List
import json
import torch
import elasticsearch
from elasticsearch import Elasticsearch
from tqdm import tqdm
from utils import get_es_client
from config import INDEX_NAME_EMBEDDING
from sentence_transformers import SentenceTransformer


def index_data(documents: List[dict], model: SentenceTransformer) -> None:
    es = get_es_client(max_retries=5, sleep_time=5)
    _ = _create_index(es=es)
    _ = _index_document(es=es, documents=documents, model=model)
    pprint(f"Indexed {len(documents)} documents into index '{INDEX_NAME_EMBEDDING}'.")

def _create_index(es: Elasticsearch) -> dict:
    index_name = INDEX_NAME_EMBEDDING
    _ = es.indices.delete(index=index_name, ignore_unavailable=True)
    return es.indices.create(
        index=index_name,
        mappings = {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    
                },
            }
        }
    )
    

def _index_document(es: Elasticsearch, documents: List[dict], model=SentenceTransformer) -> dict:
    operations = []
    index_name = INDEX_NAME_EMBEDDING
    for document in tqdm(documents, total=len(documents), desc="Indexing documents"):
        operations.append({'index': {'_index': index_name}})
        operations.append({
            **document,
            'embedding': model.encode(document['explanation'])
        })
    return es.bulk(operations=operations)

if __name__ == "__main__":
    with open('../../data/apod.json') as f:
        documents = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    index_data(documents=documents, model=model)
    