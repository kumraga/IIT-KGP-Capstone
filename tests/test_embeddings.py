"""Person 2 tests for the shared embedding API client (offline: httpx.MockTransport)."""

import json
import os

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from verisheet.config import Settings, get_settings
from verisheet.embeddings import (
    MAX_TEXTS_PER_REQUEST,
    EmbeddingAPIEncoder,
    EmbeddingServiceError,
    create_encoder,
    embedding_provider,
)
from verisheet.retrieval import DefinitionRetriever, retrieve_context

BASE = "https://embeddings.test/v1"
KEY = "test-key-should-never-appear"


@pytest.fixture(autouse=True)
def _offline_embedding_provider(monkeypatch):
    """Keep tests offline and deterministic even when .env configures the embedding API."""
    from verisheet.config import get_settings

    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def fake_vector(text: str, dim: int = 384) -> list[float]:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.normal(size=dim).tolist()


def make_handler(log: list, failures: list[int] | None = None, reverse: bool = False):
    """Fake /v1/embeddings: records requests, optionally fails first with given statuses."""
    failures = list(failures or [])

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        log.append({"url": str(request.url), "auth": request.headers.get("authorization"), **body})
        if failures:
            return httpx.Response(failures.pop(0), json={"error": "server at capacity"})
        data = [{"object": "embedding", "index": i, "embedding": fake_vector(t)}
                for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data[::-1] if reverse else data,
                                         "model": body["model"]})

    return handler


def encoder(log, **kwargs) -> EmbeddingAPIEncoder:
    sleeps = kwargs.pop("sleeps", [])
    return EmbeddingAPIEncoder(
        BASE, KEY, transport=httpx.MockTransport(make_handler(log, **kwargs)),
        sleep=sleeps.append,
    )


def test_api_encoder_batches_and_preserves_order():
    log: list = []
    texts = [f"chunk {i}" for i in range(40)]
    vectors = encoder(log, reverse=True).encode(texts, normalize_embeddings=True)

    assert vectors.shape == (40, 384) and vectors.dtype == np.float32
    assert [len(r["input"]) for r in log] == [16, 16, 8]
    assert all(len(r["input"]) <= MAX_TEXTS_PER_REQUEST for r in log)
    assert log[0]["url"] == f"{BASE}/embeddings"
    assert log[0]["model"] == "bge-small-en-v1.5"
    assert log[0]["auth"] == f"Bearer {KEY}"
    # Server returned each batch reversed; client reorders by `index`.
    assert np.allclose(vectors[5], fake_vector("chunk 5"))


def test_api_encoder_retries_capacity_errors_with_backoff():
    log: list = []
    sleeps: list = []
    vectors = encoder(log, failures=[503, 429], sleeps=sleeps).encode(["a", "b"])

    assert vectors.shape == (2, 384)
    assert len(log) == 3 and sleeps == [1.0, 2.0]


def test_api_encoder_gives_up_after_max_attempts_without_leaking_key():
    log: list = []
    with pytest.raises(EmbeddingServiceError) as info:
        encoder(log, failures=[503] * 10).encode(["a"])
    assert len(log) == 4
    assert "503" in str(info.value) and KEY not in str(info.value)


@pytest.mark.parametrize("status, match", [(401, "EMBEDDING_API_KEY"), (400, "HTTP 400")])
def test_api_encoder_does_not_retry_client_errors(status, match):
    log: list = []
    with pytest.raises(EmbeddingServiceError, match=match) as info:
        encoder(log, failures=[status]).encode(["a"])
    assert len(log) == 1 and KEY not in str(info.value)


def test_api_encoder_rejects_malformed_responses():
    def short(request):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

    client = EmbeddingAPIEncoder(BASE, KEY, transport=httpx.MockTransport(short))
    with pytest.raises(EmbeddingServiceError, match="2 inputs"):
        client.encode(["a", "b"])

    def garbage(request):
        return httpx.Response(200, text="<html>proxy error</html>")

    client = EmbeddingAPIEncoder(BASE, KEY, transport=httpx.MockTransport(garbage))
    with pytest.raises(EmbeddingServiceError, match="unexpected response"):
        client.encode(["a"])


def test_api_encoder_requires_config_and_existing_ca_cert(tmp_path):
    with pytest.raises(EmbeddingServiceError, match="EMBEDDING_API_BASE"):
        EmbeddingAPIEncoder("", KEY)
    with pytest.raises(EmbeddingServiceError, match="EMBEDDING_API_KEY"):
        EmbeddingAPIEncoder(BASE, "")
    with pytest.raises(EmbeddingServiceError, match="CA certificate not found"):
        EmbeddingAPIEncoder(BASE, KEY, ca_cert=tmp_path / "missing.crt")


def test_provider_selection_and_model_guard():
    local = Settings(_env_file=None, embedding_provider="auto")
    api = Settings(_env_file=None, embedding_provider="auto", embedding_api_base=BASE,
                   embedding_api_key=SecretStr(KEY))
    assert embedding_provider(local) == "local"
    assert embedding_provider(api) == "api"
    assert embedding_provider(Settings(_env_file=None, embedding_provider="local",
                                       embedding_api_base=BASE,
                                       embedding_api_key=SecretStr(KEY))) == "local"
    assert KEY not in repr(api)  # SecretStr keeps the key out of logs/reprs

    assert isinstance(create_encoder("BAAI/bge-small-en-v1.5", api), EmbeddingAPIEncoder)
    with pytest.raises(ValueError, match="same model"):
        create_encoder("BAAI/bge-base-en-v1.5", api)


def test_retriever_end_to_end_with_api_encoder(tmp_path):
    log: list = []
    retriever = DefinitionRetriever(model_name="BAAI/bge-small-en-v1.5", index_dir=str(tmp_path))
    retriever._encoder = encoder(log)
    docs = [{"item_id": f"d{i}", "source_path": f"docs/x.md#d{i}", "title": f"T{i}",
             "content": f"rule number {i}"} for i in range(20)]
    retriever.build_index(docs)
    context = retriever.search("rule number 3", top_k=3)

    assert len(context.items) == 3
    assert [len(r["input"]) for r in log] == [16, 4, 1]  # 20 docs in 2 batches + 1 query
    assert log[-1]["input"][0].startswith("Represent this sentence")  # BGE query prefix kept


def test_retrieve_context_degrades_when_embedding_api_down(tmp_path):
    retriever = DefinitionRetriever(model_name="BAAI/bge-small-en-v1.5", index_dir=str(tmp_path))
    retriever._encoder = encoder([])
    retriever.build_index([{"item_id": "a", "source_path": "docs/a.md", "title": "A",
                            "content": "margin rule"}])
    retriever._encoder = encoder([], failures=[503] * 10)

    context = retrieve_context("margin", None, retriever=retriever)
    assert context.items == [] and context.corpus_version == "unavailable"
    assert context.warnings[0].startswith("RETRIEVAL_UNAVAILABLE") and KEY not in context.warnings[0]


@pytest.mark.skipif(os.environ.get("VERISHEET_LIVE_EMBEDDING_TEST") != "1",
                    reason="live API test; set VERISHEET_LIVE_EMBEDDING_TEST=1 and EMBEDDING_* vars")
def test_live_embedding_api_smoke(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "api")
    get_settings.cache_clear()
    client = create_encoder("BAAI/bge-small-en-v1.5", get_settings())
    vectors = client.encode(["Margin is sum(Profit) / sum(Sales).", "how is margin calculated"])
    assert vectors.shape == (2, 384)
