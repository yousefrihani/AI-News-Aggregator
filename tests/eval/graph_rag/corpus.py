"""Fixed corpus and golden questions for the Graph RAG evaluation.

The corpus is small on purpose: it is a regression fixture, not a benchmark.
Embeddings are computed with a deterministic bag-of-words encoder so the
evaluation runs offline and never downloads a model. Swapping in
``FastEmbedEmbedding`` later only changes the ranking, not the metric plumbing.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from Articles.graph.base import Chunk, Entity, Mention, Relation
from Articles.graph.memory import InMemoryGraphStore
from Articles.Models import Article

__all__ = [
    "ARTICLES",
    "BagOfWordsEmbedding",
    "GOLDEN",
    "GoldenQuestion",
    "build_store",
    "chunk_ids",
    "supporting_article_ids",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PUBLISHED = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ArticleSpec:
    """A fixed article with explicit retrieval chunks."""

    id: str
    title: str
    chunks: tuple[str, ...]

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Return the deterministic chunk identifiers for this article."""
        return tuple(f"{self.id}-c{ordinal}" for ordinal in range(len(self.chunks)))


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    """A golden question with its supporting evidence and recorded answer."""

    question: str
    relevant_chunk_ids: tuple[str, ...]
    supporting_article_ids: tuple[str, ...]
    answer: str


ARTICLES: tuple[ArticleSpec, ...] = (
    ArticleSpec(
        "a01",
        "OpenAI releases GPT-5",
        (
            "OpenAI released GPT-5, a multimodal model with improved reasoning and a "
            "400k token context window.",
            "GPT-5 is available through the OpenAI API and powers new agent workflows "
            "in the Responses API.",
        ),
    ),
    ArticleSpec(
        "a02",
        "Nvidia unveils Blackwell Ultra",
        (
            "Nvidia announced Blackwell Ultra GPUs with 288GB of HBM3e memory for large "
            "model training.",
            "Blackwell Ultra systems ship in 2027 and target data centers running "
            "trillion parameter models.",
        ),
    ),
    ArticleSpec(
        "a03",
        "EU AI Act enforcement begins",
        (
            "The European Union began enforcing the AI Act, requiring risk assessments "
            "for high risk systems.",
            "Companies face fines up to 7 percent of global revenue for prohibited AI "
            "practices under the AI Act.",
        ),
    ),
    ArticleSpec(
        "a04",
        "Anthropic publishes Claude safety report",
        (
            "Anthropic published a safety report for Claude, describing red teaming and "
            "constitutional training.",
            "Claude deployments now include automated jailbreak detection and usage "
            "policies for enterprise customers.",
        ),
    ),
    ArticleSpec(
        "a05",
        "Google DeepMind introduces Gemini 3",
        (
            "Google DeepMind introduced Gemini 3 with a mixture of experts architecture "
            "and long context retrieval.",
            "Gemini 3 benchmarks show gains in coding, mathematics, and multimodal "
            "reasoning tasks.",
        ),
    ),
    ArticleSpec(
        "a06",
        "Meta releases Llama 4",
        (
            "Meta released Llama 4 under a permissive community license for research and "
            "commercial use.",
            "Llama 4 weights are available on Hugging Face and run on consumer GPUs with "
            "quantization.",
        ),
    ),
    ArticleSpec(
        "a07",
        "New AI chip export controls",
        (
            "New export controls restrict advanced AI chip sales to certain countries, "
            "affecting data center plans.",
            "Chipmakers are seeking licenses while cloud providers reroute GPU capacity "
            "to compliant regions.",
        ),
    ),
    ArticleSpec(
        "a08",
        "Robotics startup trains manipulation model",
        (
            "A robotics startup trained a foundation model on manipulation data from warehouses.",
            "The robot policy generalizes to new objects and improves grasping success by "
            "30 percent.",
        ),
    ),
    ArticleSpec(
        "a09",
        "AI workloads drive record energy demand",
        (
            "Data centers powering AI workloads are driving record electricity demand and "
            "grid upgrades.",
            "Utilities plan new gas plants and long duration storage to meet AI energy "
            "demand by 2030.",
        ),
    ),
    ArticleSpec(
        "a10",
        "AI diagnostic tool beats radiologists in trial",
        (
            "A clinical trial found an AI diagnostic tool detected early cancers more "
            "accurately than radiologists.",
            "Hospitals plan to deploy the diagnostic model after regulatory clearance and "
            "clinician training.",
        ),
    ),
)

GOLDEN: tuple[GoldenQuestion, ...] = (
    GoldenQuestion(
        "What context window does GPT-5 support?",
        ("a01-c0",),
        ("a01",),
        "GPT-5 supports a 400k token context window [a01].",
    ),
    GoldenQuestion(
        "How can developers access GPT-5 through the API?",
        ("a01-c1",),
        ("a01",),
        "GPT-5 is available through the OpenAI API and the Responses API [a01].",
    ),
    GoldenQuestion(
        "How much HBM3e memory does Blackwell Ultra have?",
        ("a02-c0",),
        ("a02",),
        "Blackwell Ultra GPUs carry 288GB of HBM3e memory [a02].",
    ),
    GoldenQuestion(
        "When do Blackwell Ultra systems ship?",
        ("a02-c1",),
        ("a02",),
        "Blackwell Ultra systems ship in 2027 for trillion parameter model data centers [a02].",
    ),
    GoldenQuestion(
        "What does the EU AI Act require for high risk systems?",
        ("a03-c0",),
        ("a03",),
        "The AI Act requires risk assessments for high risk systems [a03].",
    ),
    GoldenQuestion(
        "What fines does the AI Act impose for prohibited practices?",
        ("a03-c1",),
        ("a03",),
        "Fines reach up to 7 percent of global revenue for prohibited AI practices [a03].",
    ),
    GoldenQuestion(
        "What safety training does Claude use?",
        ("a04-c0",),
        ("a04",),
        "Anthropic describes red teaming and constitutional training for Claude [a04].",
    ),
    GoldenQuestion(
        "How does Claude detect jailbreak attempts?",
        ("a04-c1",),
        ("a04",),
        "Claude deployments include automated jailbreak detection for enterprise customers [a04].",
    ),
    GoldenQuestion(
        "What architecture does Gemini 3 use?",
        ("a05-c0",),
        ("a05",),
        "Gemini 3 uses a mixture of experts architecture with long context retrieval [a05].",
    ),
    GoldenQuestion(
        "Which benchmarks improved for Gemini 3?",
        ("a05-c1",),
        ("a05",),
        "Gemini 3 improved coding, mathematics, and multimodal reasoning benchmarks [a05].",
    ),
    GoldenQuestion(
        "What license does Llama 4 use?",
        ("a06-c0",),
        ("a06",),
        "Llama 4 uses a permissive community license for research and commercial use [a06].",
    ),
    GoldenQuestion(
        "Where are Llama 4 weights available?",
        ("a06-c1",),
        ("a06",),
        "Llama 4 weights are on Hugging Face and run on consumer GPUs with quantization [a06].",
    ),
    GoldenQuestion(
        "Which chip sales do export controls restrict?",
        ("a07-c0",),
        ("a07",),
        "Export controls restrict advanced AI chip sales to certain countries [a07].",
    ),
    GoldenQuestion(
        "How are cloud providers rerouting GPU capacity?",
        ("a07-c1",),
        ("a07",),
        "Cloud providers reroute GPU capacity to compliant regions while seeking licenses [a07].",
    ),
    GoldenQuestion(
        "What data was the robotics foundation model trained on?",
        ("a08-c0",),
        ("a08",),
        "The robotics foundation model was trained on warehouse manipulation data [a08].",
    ),
    GoldenQuestion(
        "How much did grasping success improve for the robot policy?",
        ("a08-c1",),
        ("a08",),
        "The robot policy improved grasping success by 30 percent and generalizes to new "
        "objects [a08].",
    ),
    GoldenQuestion(
        "What is driving electricity demand from data centers?",
        ("a09-c0",),
        ("a09",),
        "Data centers powering AI workloads are driving record electricity demand [a09].",
    ),
    GoldenQuestion(
        "How do utilities plan to meet AI energy demand?",
        ("a09-c1",),
        ("a09",),
        "Utilities plan new gas plants and long duration storage to meet AI energy demand "
        "by 2030 [a09].",
    ),
    GoldenQuestion(
        "How accurate is the AI diagnostic tool for early cancers?",
        ("a10-c0",),
        ("a10",),
        "The AI diagnostic tool detected early cancers more accurately than radiologists [a10].",
    ),
    GoldenQuestion(
        "What must hospitals do before deploying the diagnostic model?",
        ("a10-c1",),
        ("a10",),
        "Hospitals need regulatory clearance and clinician training before deploying the "
        "diagnostic model [a10].",
    ),
    GoldenQuestion(
        "How do GPT-5 and Gemini 3 handle long context?",
        ("a01-c0", "a05-c0"),
        ("a01", "a05"),
        "GPT-5 supports a 400k token context window [a01], while Gemini 3 uses long "
        "context retrieval [a05].",
    ),
)

# Entity mentions connect chunks so depth-1 expansion has real edges to walk.
_ENTITIES: tuple[Entity, ...] = (
    Entity("e-openai", "OpenAI", "organization"),
    Entity("e-gpt5", "GPT-5", "product"),
    Entity("e-nvidia", "Nvidia", "organization"),
    Entity("e-blackwell", "Blackwell Ultra", "product"),
    Entity("e-eu", "European Union", "organization"),
    Entity("e-ai-act", "AI Act", "regulation"),
    Entity("e-anthropic", "Anthropic", "organization"),
    Entity("e-claude", "Claude", "product"),
    Entity("e-google", "Google DeepMind", "organization"),
    Entity("e-gemini", "Gemini 3", "product"),
    Entity("e-meta", "Meta", "organization"),
    Entity("e-llama", "Llama 4", "product"),
)
_MENTIONS: tuple[Mention, ...] = (
    Mention("a01-c0", "e-openai"),
    Mention("a01-c0", "e-gpt5"),
    Mention("a01-c1", "e-openai"),
    Mention("a01-c1", "e-gpt5"),
    Mention("a02-c0", "e-nvidia"),
    Mention("a02-c0", "e-blackwell"),
    Mention("a02-c1", "e-nvidia"),
    Mention("a02-c1", "e-blackwell"),
    Mention("a03-c0", "e-eu"),
    Mention("a03-c0", "e-ai-act"),
    Mention("a03-c1", "e-eu"),
    Mention("a03-c1", "e-ai-act"),
    Mention("a04-c0", "e-anthropic"),
    Mention("a04-c0", "e-claude"),
    Mention("a04-c1", "e-anthropic"),
    Mention("a04-c1", "e-claude"),
    Mention("a05-c0", "e-google"),
    Mention("a05-c0", "e-gemini"),
    Mention("a05-c1", "e-google"),
    Mention("a05-c1", "e-gemini"),
    Mention("a06-c0", "e-meta"),
    Mention("a06-c0", "e-llama"),
    Mention("a06-c1", "e-meta"),
    Mention("a06-c1", "e-llama"),
)
_RELATIONS: tuple[Relation, ...] = (
    Relation("e-openai", "e-gpt5", "released", confidence=0.95, evidence_chunk_id="a01-c0"),
    Relation("e-nvidia", "e-blackwell", "announced", confidence=0.9, evidence_chunk_id="a02-c0"),
    Relation("e-eu", "e-ai-act", "enforces", confidence=0.9, evidence_chunk_id="a03-c0"),
    Relation("e-anthropic", "e-claude", "publishes_report_on", confidence=0.85),
    Relation("e-google", "e-gemini", "introduced", confidence=0.9, evidence_chunk_id="a05-c0"),
    Relation("e-meta", "e-llama", "released", confidence=0.9, evidence_chunk_id="a06-c0"),
)


class BagOfWordsEmbedding:
    """Deterministic token-overlap embeddings for the evaluation fixture.

    Tokens are hashed into a fixed-size vector with signed counts, which keeps
    shared vocabulary correlated without a model download. The encoder is only
    used by the evaluation harness.

    Args:
        dim: Vector dimension. Must be greater than zero.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim < 1:
            raise ValueError("dim must be at least 1")
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        """Encode one text as a normalized signed token histogram."""
        values = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.casefold()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:8], "big") % self.dim
            sign = 1.0 if digest[8] & 1 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            return values
        return [value / norm for value in values]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        return [self._vector(text) for text in texts]

    async def aclose(self) -> None:
        """Release resources; none are owned."""


def _article(spec: ArticleSpec) -> Article:
    """Build a normalized article from a spec."""
    return Article(
        id=spec.id,
        title=spec.title,
        url=f"https://example.com/{spec.id}",
        source="EvalFixture",
        published_at=_PUBLISHED,
        summary=spec.chunks[0],
        topics=("ai",),
    )


async def build_store(
    embedder: BagOfWordsEmbedding | None = None,
) -> InMemoryGraphStore:
    """Build the fixed evaluation corpus in an in-memory store.

    Args:
        embedder: Encoder used for chunk vectors. Defaults to a fresh
            :class:`BagOfWordsEmbedding`.

    Returns:
        A populated store ready for retrieval.
    """
    encoder = embedder or BagOfWordsEmbedding()
    texts = [text for spec in ARTICLES for text in spec.chunks]
    vectors = await encoder.embed(texts)
    chunks: list[Chunk] = []
    cursor = 0
    for spec in ARTICLES:
        for ordinal, chunk_id in enumerate(spec.chunk_ids):
            chunks.append(
                Chunk(
                    id=chunk_id,
                    article_id=spec.id,
                    ordinal=ordinal,
                    text=spec.chunks[ordinal],
                    embedding=tuple(vectors[cursor]),
                )
            )
            cursor += 1
    store = InMemoryGraphStore()
    await store.upsert_articles([_article(spec) for spec in ARTICLES], chunks)
    await store.upsert_entities(list(_ENTITIES), list(_RELATIONS), list(_MENTIONS))
    return store


def chunk_ids() -> frozenset[str]:
    """Return every chunk identifier in the fixed corpus."""
    return frozenset(chunk_id for spec in ARTICLES for chunk_id in spec.chunk_ids)


def supporting_article_ids() -> frozenset[str]:
    """Return every article identifier in the fixed corpus."""
    return frozenset(spec.id for spec in ARTICLES)
