"""Backward-compatible imports for the token-based context curriculum."""

from .context_curriculum import (
    ContextCurriculumConfig,
    ContextCurriculumState,
    ContextStage,
    ContextTransition,
    ProgressiveContextCollator,
    ProgressiveContextCurriculum,
    build_context_loader,
    truncate_batch_to_context,
)

ContextCurriculum = ProgressiveContextCurriculum

__all__ = [
    "ContextCurriculum",
    "ContextCurriculumConfig",
    "ContextCurriculumState",
    "ContextStage",
    "ContextTransition",
    "ProgressiveContextCollator",
    "ProgressiveContextCurriculum",
    "build_context_loader",
    "truncate_batch_to_context",
]
