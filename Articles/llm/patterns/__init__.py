"""LLM patterns: typed strategies over the route-based router.

Patterns own prompt shaping and output validation. They never know a provider
or model name; every call goes through a route such as ``summarize`` or
``extract_graph``.
"""

from Articles.llm.patterns.base import (
    CompletionRouter,
    Pattern,
    PatternContext,
    PatternRegistry,
    SingleShotPattern,
    parse_json_object,
)
from Articles.llm.patterns.classify import Classification, ClassifyPattern, parse_classification
from Articles.llm.patterns.extract_graph import (
    GRAPH_EXTRACTION_SCHEMA,
    GRAPH_EXTRACTION_SCHEMA_REF,
    ExtractedEntity,
    ExtractedMention,
    ExtractedRelation,
    ExtractGraphPattern,
    GraphExtraction,
    parse_graph_extraction,
)
from Articles.llm.patterns.summarize import SummarizePattern

__all__ = [
    "Classification",
    "ClassifyPattern",
    "CompletionRouter",
    "ExtractGraphPattern",
    "ExtractedEntity",
    "ExtractedMention",
    "ExtractedRelation",
    "GRAPH_EXTRACTION_SCHEMA",
    "GRAPH_EXTRACTION_SCHEMA_REF",
    "GraphExtraction",
    "Pattern",
    "PatternContext",
    "PatternRegistry",
    "SingleShotPattern",
    "SummarizePattern",
    "parse_classification",
    "parse_graph_extraction",
    "parse_json_object",
]
