# Elasticsearch Practice

This repo is forked from ImadSaddik where I work step by step with his ElasticSearch Project. The goal of this project was to gain exposure to Elasticsearch and its different features.

## Repository contents

#### Core Features
- Text Search: Multi-field search across document titles and explanations
- N-gram Analysis: Enhanced search capabilities with edge n-gram tokenization
- Semantic Search: Vector-based search using sentence embeddings
- FastAPI Backend: RESTful API endpoints for search functionality
- Year-based Filtering: Filter search results by date ranges

#### Semantic Search
- Used sentence-transformers to create embedded vectors for each image explanation
- Use the semantic similarity scores to compare to user qeuries

#### N-Grams Search
- Improved on regular standard text matching with partial matching & autocomplete


## Prerequisites
- Elasticsearch was ran locally on a Docker container:
- docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.15.0
- Sentence-Transformers was also used and installed in order to create the embeddings
- Built on Vue.js for frontend


