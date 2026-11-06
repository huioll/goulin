# -*- coding: utf-8 -*-
"""向量化后端。

两个实现，同一个接口：
  * OpenAIEmbedder —— 走 OpenAI 兼容的 /embeddings 接口（WorkBuddy 若支持就直接用）。
  * LocalHashEmbedder —— 本地特征哈希（中文按字 + 二元组，英文按词），SHA/blake2b 映射到定长向量。
    不联网、可复现、零依赖，保证整套记忆体系在任何环境都能跑起来。

provider="auto" 时会先试着用接口，失败自动降级到本地，并把这个降级写进 trace。
检索用暴力余弦（SQLite 存 BLOB）——数据量在万级以内够用；换 Chroma/pgvector 只需替换 LongTermMemory 的检索部分。
"""

import hashlib
import math
import re
from array import array

from . import config

_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-z0-9_]+")


def cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def to_blob(vec) -> bytes:
    return array("f", [float(x) for x in vec]).tobytes()


def from_blob(blob) -> list:
    if not blob:
        return []
    arr = array("f")
    arr.frombytes(blob)
    return list(arr)


class LocalHashEmbedder:
    """确定性本地向量：字/二元组/词 → blake2b → 有符号累加 → L2 归一化。"""

    name = "local-hash"

    def __init__(self, dim: int = config.LOCAL_EMBED_DIM):
        self.dim = dim

    @staticmethod
    def _features(text: str):
        text = (text or "").lower()
        feats = []
        for word in _WORD.findall(text):
            feats.append("w:" + word)
        chars = _CJK_CHAR.findall(text)
        for ch in chars:
            feats.append("u:" + ch)
        for a, b in zip(chars, chars[1:]):
            feats.append("b:" + a + b)
        # 数字也单独作为特征（学号、分数很重要）
        for num in re.findall(r"\d+(?:\.\d+)?", text):
            feats.append("n:" + num)
        return feats

    def encode(self, text: str) -> list:
        vec = [0.0] * self.dim
        for feat in self._features(text):
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class OpenAIEmbedder:
    """调用 OpenAI 兼容的 embeddings 接口。"""

    def __init__(self, client, model: str = config.EMBED_MODEL):
        self.client = client
        self.model = model
        self.dim = 0

    def encode(self, text: str) -> list:
        resp = self.client.embeddings.create(model=self.model, input=text)
        vec = list(resp.data[0].embedding)
        self.dim = len(vec)
        return vec


class Embedder:
    """统一入口：负责 provider 选择、维度记录、降级与失败兜底。"""

    def __init__(self, client=None, tracer=None, provider: str = None):
        self.tracer = tracer
        self.provider = (provider or config.EMBED_PROVIDER).lower()
        self._local = LocalHashEmbedder()
        self._remote = OpenAIEmbedder(client) if (client and self.provider != "local") else None
        self.backend = "local-hash"
        self.dim = self._local.dim
        if self._remote:
            self.backend = f"openai:{self._remote.model}"
            self.dim = 0   # 首次调用后才知道

    def encode(self, text: str) -> list:
        if self._remote and self.provider in {"auto", "openai"}:
            try:
                vec = self._remote.encode(text)
                self.backend = f"openai:{self._remote.model}"
                self.dim = self._remote.dim
                return vec
            except Exception as exc:
                if self.provider == "openai":
                    raise
                if self.tracer:
                    self.tracer.event("embed.fallback", reason=f"{type(exc).__name__}: {exc}",
                                      fallback="local-hash")
                self._remote = None
        vec = self._local.encode(text)
        self.backend = "local-hash"
        self.dim = self._local.dim
        return vec

    def encode_many(self, texts) -> list:
        return [self.encode(t) for t in texts]
