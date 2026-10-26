
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