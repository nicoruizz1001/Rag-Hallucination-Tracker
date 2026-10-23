import importlib

import pytest

CORE_LIBRARIES = [
    "datasets",
    "pandas",
    "numpy",
    "sentence_transformers",
    "faiss",
    "rank_bm25",
    "ollama",
    "matplotlib",
    "sklearn",
    "scipy",
    "tqdm",
]

PROJECT_MODULES = [
    "src.prepare_data",
    "src.retrieval",
    "src.generation",
    "src.evaluation",
    "src.experiments",
    "src.plots",
]


@pytest.mark.parametrize("name", CORE_LIBRARIES + PROJECT_MODULES)
def test_importable(name):
    importlib.import_module(name)
