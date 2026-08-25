"""Unified human-model classes.

Entry point for new code. Import from here instead of the legacy modules:

    from final_models import HumanMPPI, HumanCrocoddyl
"""
from .human_base import HumanBase
from .human_crocoddyl import HumanCrocoddyl
from .human_mppi import HumanMPPI

__all__ = ["HumanBase", "HumanMPPI", "HumanCrocoddyl"]
