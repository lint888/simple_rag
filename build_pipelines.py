from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter



docs = [Document(content="tom is a student living in frankfurt germany"),
        Document(content="tom is going to visit the frankfurt mesuem next sunday")]

# ingestion data
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()
document_store = InMemoryDocumentStore()
document_writer = DocumentWriter(document_store)


p = Pipeline()

p.add_component("doc_embedder", doc_embedder)
p.add_component("writer", document_writer)

p.connect("doc_embedder", "writer")

p.run({"doc_embedder": {"documents": docs}})


rag_p = Pipeline()
query_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
retriever = InMemoryEmbeddingRetriever(document_store)



from haystack.components.builders import PromptBuilder
from haystack.dataclasses import ChatMessage

template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""



prompt_builder = PromptBuilder(template=template)
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack_integrations.components.generators.ollama import OllamaGenerator

generator = OllamaGenerator(model="llama2:7b",
                            url = "http://localhost:11434")

rag_p.add_component("text_embedder", query_embedder)
rag_p.add_component("retriever", retriever)
rag_p.add_component("prompt_builder", prompt_builder)
rag_p.add_component("generator", generator)

# Now, connect the components to each other
rag_p.connect("text_embedder.embedding", "retriever.query_embedding")
rag_p.connect("retriever", "prompt_builder")
rag_p.connect("prompt_builder.prompt", "generator.prompt")

question = "where does tom live and what will tom do next sunday"

response = rag_p.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

print(response["generator"]["replies"][0])

