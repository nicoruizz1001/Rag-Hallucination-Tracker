
from datasets import load_dataset
import pandas as pd
import json
import os
import hashlib
import json
from pathlib import Path

dataset = load_dataset(
    "rajpurkar/squad_v2",
    split="validation"
)

shuffled = dataset.shuffle(seed=42)

sample = shuffled.select(range(100))


questions = []

for row in sample:
    questions.append({
        "id": row["id"],
        "question": row["question"],
        "context": row["context"],
        "answers": json.dumps(row["answers"]["text"]),
        "is_answerable": len(row["answers"]["text"]) > 0
    })


df = pd.DataFrame(questions)


os.makedirs("data", exist_ok=True)

df.to_csv("data/validation_questions.csv", index=False)

assert len(sample) == 100

print("Total questions:", len(df))
print("Answerable:", df["is_answerable"].sum())
print("Unanswerable:", (~df["is_answerable"]).sum())
print(df[["question", "is_answerable"]].head())

records = []

for row in sample:
    reference_answers = row["answers"]["text"]

    records.append({
        "question_id": row["id"],
        "question": row["question"],
        "source_passage": row["context"],
        "reference_answers": json.dumps(reference_answers),
        "is_answerable": len(reference_answers) > 0,
    })

df = pd.DataFrame(records)

assert len(df) == 100
assert df["question_id"].is_unique
assert df["is_answerable"].any()
assert (~df["is_answerable"]).any()

# Save inside the project's data folder.
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

output_path = data_dir / "questions.csv"
df.to_csv(output_path, index=False)

print("\nTotal questions:", len(df))
print("Answerable:", df["is_answerable"].sum())
print("Unanswerable:", (~df["is_answerable"]).sum())
print("Saved to:", output_path)

print(df[["question_id", "question", "is_answerable"]].head())



def get_context_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Connect each question to its original source passage.
df["gold_context_id"] = df["source_passage"].apply(get_context_id)

# Use the column name specified in the project guide.
df = df.rename(columns={"reference_answers": "gold_answer"})

# Start with every sampled question's source passage.
# A dictionary prevents duplicate passages.
documents = {}

for text in df["source_passage"]:
    document_id = get_context_id(text)
    documents[document_id] = text

# Add other passages as distractors until we have 500 total.
target_size = 500

for row in shuffled:
    if len(documents) >= target_size:
        break

    text = row["context"]
    document_id = get_context_id(text)
    documents[document_id] = text

documents_df = pd.DataFrame(
    [
        {"document_id": document_id, "text": text}
        for document_id, text in documents.items()
    ]
)

# Check that the corpus is complete and has no duplicates.
assert len(documents_df) == target_size
assert documents_df["document_id"].is_unique
assert documents_df["text"].is_unique
assert df["gold_context_id"].isin(documents_df["document_id"]).all()

# Save the updated questions and passage corpus.
df.to_csv(data_dir / "questions.csv", index=False)
documents_df.to_csv(data_dir / "documents.csv", index=False)

print("\nSaved questions:", len(df))
print("Saved unique passages:", len(documents_df))
print("Every question's source passage is in the corpus.")