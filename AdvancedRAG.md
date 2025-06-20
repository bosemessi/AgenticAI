# Four Categories of Advanced RAGs -> Pre-retrieval and Indexing, Retrieval, Post-retrieval, Generation

## Pre-retrieval and Indexing

### Issues with data
- information density - lots of irrelevant information combined with relevant text
- repetition/duplication of content or meaning/intent of content

### Solutions
- increase info density with LLMs
    - raw text -> LLM ->summarise/key word or fact extraction/data cleaning/labelling -> dense info with tags/categories/key facts
- deduplication
    - cluster similar text chunks (in embedding space) and use LLM to generate deduplicate chunk
- hypothetical question index
    - pre-prepare questions using LLM for the chunks. 
    - match user query with the most similar pre-prepared questions
    - return the chunk to which the most similar pre-prepared questions point to

### Techniques involved
- chunking
    - simple chunking
    - semantic chunking - more expensive but more relevant

## Retrieval

### Solutions
- optimise query through LLMs
    - pass query through fine-tuned LLM to restructure into a format better understood by LLMs. Add metadata. Remove irrelevant bits.
- hierarchical index retrieval
    - hierarchical chunking - parent chunk nodes - more fine-grained subchunks
    - semantic search among parent nodes, then with fine-grained subchunks
    - helps delaing with removal of irrelevant info
- HyDE - hypothetical document embeddings
    - this time, ask LLM to generate sample answers to the user query
    - find best-matching chunks to these LLM generated answer(s), feed to LLM to generate final answer.
- Query-routing or RAG decider
    - pass query through LLM to decide what retrival or generation system is best-suited to answer
    - for eg. if retrieval is at all required or not
- Self-Query
    - based on initial user query, LLM generates further follow-up queries
- Hybrid Search
    - keyword search + semantic search combined
- Graph search
    - nodes as entities and edges as relationships
- Finetuning embedding model
    - finetuning model for particualr use-cases.

## Post-Retrieval

### Solutions
- Reranking
    - reranking chunks based on relevancy
- Contextual prompt compression
    - use LLM to pass the original chunks and produce compressed chunks
- corrective RAG (CRAG)
    - use a corrector LLM to rank and filter out chunks (correct, incorrect, ambiguous)
- Query expansion
    - when query is too specific, expand out with synonyms/related terms/alternative phrases

## Generation

### Solutions
- Chain-of-thought prompting
    - teaching model to follow logical sequence of steps (sub-queries) to make response more relevant for specific queries
- Self-RAG
    - multiple feedback rounds of response generation using LLM
- Fine-tuning LLMs
    - for specific domain
- Natural Language Inference
    - remove irrelevant context with natural language inference model


Source used : https://www.falkordb.com/blog/advanced-rag/
