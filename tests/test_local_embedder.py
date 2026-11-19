"""実モデルを使わず、ローカル推論の再利用時の振る舞いを確認します。"""

from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from mediasearch.embedders.base import TextPart
from mediasearch.embedders.local import LocalEmbedder, _PreparedWindow


def test_text_cache_keeps_exact_queries_and_does_not_share_mutable_arrays():
    embedder = LocalEmbedder.__new__(LocalEmbedder)
    calls = []

    def encode(texts, prompt):
        calls.append((list(texts), prompt))
        return np.full((len(texts), 2), len(calls), dtype=np.float32)

    embedder._encode_texts = encode
    embedder._query_cache = lru_cache(maxsize=256)(encode)
    original = embedder.embed_texts(["青い車"])
    original[:] = 0
    np.testing.assert_array_equal(embedder.embed_texts(["青い車"]), [[1, 1]])
    embedder.embed_texts([" 青い車"])
    embedder.embed_texts(["青い車"], kind="document")
    embedder.embed_texts(["青い車", "赤い車"])
    assert calls == [
        (["青い車"], "query"),
        ([" 青い車"], "query"),
        (["青い車"], "document"),
        (["青い車", "赤い車"], "query"),
    ]


def test_prepared_video_batch_falls_back_when_interleaved_with_stream_windows():
    embedder = LocalEmbedder.__new__(LocalEmbedder)
    embedder._dims = 2
    embedder._run_encode = lambda fn, *, high: fn()
    embedder._model = SimpleNamespace(encode_prepared=lambda batch: np.array([[0, 1], [1, 0]], dtype=np.float32))
    calls = []

    def encode(messages, *, high, batch_size):
        calls.append([message[0]["content"][0]["text"] for message in messages])
        return np.array([[1, 0], [0, 1]], dtype=np.float32)

    embedder._encode_array = encode
    batch = SimpleNamespace(order=np.array([1, 0]))
    first = _PreparedWindow([TextPart("first")], batch, 0)
    second = _PreparedWindow([TextPart("second")], batch, 1)
    np.testing.assert_array_equal(embedder._encode_windows([first, second]), [[1, 0], [0, 1]])
    assert not calls
    embedder._encode_windows([first, [TextPart("stream")]])
    assert calls == [["first", "stream"]]
