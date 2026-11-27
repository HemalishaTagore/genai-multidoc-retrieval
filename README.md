## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

To develop a retrieval-based AI agent that can read and process multiple research articles, retrieve relevant information from them based on user queries, and generate accurate, concise, and context-aware answers by combining information from different documents.

### DESIGN STEPS:

#### STEP 1: Document Loading and Indexing
- Collect multiple research articles in PDF or text format.
- Load the documents using LlamaIndex document readers.
- Split the documents into smaller text chunks.
- Generate embeddings and store them in a vector index.

#### STEP 2: Retrieval and Response Generation
- Accept a user query.
- Retrieve the most relevant text chunks from the indexed documents.
- Pass the retrieved information to the language model.
- Generate a concise answer using the information from multiple documents.

#### STEP 3: Testing and Evaluation
- Test the agent using different types of queries.
- Check whether the answers are relevant, accurate, and concise.
- Verify whether the agent can combine information from multiple research articles.

### PROGRAM:
```python

import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini")

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

documents = SimpleDirectoryReader("data").load_data()

if not documents:
    print("No research articles found!")
    exit()

print("Number of documents loaded:", len(documents))

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
    similarity_top_k=4,
    response_mode="compact"
)

print("\nMultidocument Retrieval Agent is ready!")

queries = [
    "What are the main findings of the research articles?",
    "Compare the methodologies used in the documents.",
    "What are the limitations mentioned in the research?"
]

for query in queries:
    print("\nQuery:", query)

    response = query_engine.query(query)

    print("Answer:", response)

    print("Sources:")
    for source in response.source_nodes:
        print("-", source.node.metadata.get("file_name", "Unknown"))

```

### OUTPUT:
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/03fcb78f-b7e4-4322-965a-97c334646dfd" />


### RESULT:
Thus, a multidocument retrieval agent was designed and implemented using LlamaIndex. The agent successfully loads multiple research articles, retrieves relevant information, and synthesizes concise responses to diverse queries. Its performance can be evaluated based on relevance, accuracy, completeness, and response conciseness.
