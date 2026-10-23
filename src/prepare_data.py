"""prepare_data module for the RAG reliability benchmark."""
from datasets import load_dataset

dataset = load_dataset("rajpurkar/squad_v2")

print(dataset)
print(dataset["validation"][0])
