"""Distillation pipeline: teacher client, dataset generator, dedup."""

from src.distill.generator import (
    GeneratorConfig,
    TeacherProtocol,
    generate_examples,
    hash_record,
    load_existing_hashes,
)
from src.distill.teacher_client import (
    DEFAULT_ENDPOINTS,
    QWEN3_THINKING_MODELS,
    GenerateParams,
    RetryPolicy,
    TeacherCache,
    TeacherClient,
    TeacherError,
    cache_key,
)

__all__ = [
    "GeneratorConfig",
    "TeacherProtocol",
    "generate_examples",
    "hash_record",
    "load_existing_hashes",
    "DEFAULT_ENDPOINTS",
    "QWEN3_THINKING_MODELS",
    "GenerateParams",
    "RetryPolicy",
    "TeacherCache",
    "TeacherClient",
    "TeacherError",
    "cache_key",
]
