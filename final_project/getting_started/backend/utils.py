import time
from pprint import pprint
from elasticsearch import Elasticsearch
import elasticsearch
def get_es_client(max_retries: int=2, sleep_time: int = 1) -> Elasticsearch:
    """Create and return an Elasticsearch client."""
    i = 0
    while i < max_retries:
        try:
            es = Elasticsearch('http://localhost:9200')
            client_info = es.info()
            print("Elasticsearch client created successfully.")
            pprint(client_info)
            return es
        except Exception:
            pprint("Could not connect to Elasticsearch, retrying...")
            i += 1
            time.sleep(sleep_time)
    raise ConnectionError("Failed to connect to Elasticsearch after multiple retries.")

