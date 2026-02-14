"""Backward-compat shim. Import from app.core.dataset instead."""
from app.core.dataset import DatasetService, MergeJobManager, MergeJobStatus, MergeJob, MergeValidationResult

__all__ = ["DatasetService", "MergeJobManager", "MergeJobStatus", "MergeJob", "MergeValidationResult"]
