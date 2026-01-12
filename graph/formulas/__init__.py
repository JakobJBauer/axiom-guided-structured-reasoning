"""
Formula system for graph nodes.

Supports: Not, And, Xor, Or, Equal, In
"""

from .formula import Formula
from .operations import Not, And, Xor, Or, Equal, In

__all__ = ['Formula', 'Not', 'And', 'Xor', 'Or', 'Equal', 'In']

