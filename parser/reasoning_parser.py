"""
Parser for generating reasoning trees from reasoning text.

Parses reasoning text in the format:
<thinking>
Paragraph with [SOURCE] nodes and (TARGET : True/False) conclusion.
</thinking>
"""

import re
import sys
import os
from typing import List, Dict, Optional, Set

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph.graph import Node, Edge, Graph


class ReasoningParser:
    """
    Parse reasoning text into reasoning trees.
    
    Format:
    - Paragraphs with [NODE] citations (sources)
    - (NODE : True/False) at end indicates target node and its value
    - Edges are created from [NODE] sources to (NODE) target
    - Case-insensitive matching, but node names stored as uppercase
    - A node is never connected to itself
    """
    
    def __init__(self):
        """Initialize the parser."""
        pass
    
    def parse(self, reasoning_text: str) -> Graph:
        """
        Parse reasoning text into a reasoning tree.
        
        Args:
            reasoning_text: Text within <thinking>...</thinking> tags or raw text
        
        Returns:
            Graph representing the reasoning tree
        """
        # Extract text from <thinking> tags if present
        thinking_match = re.search(
            r'<thinking>(.*?)</thinking>',
            reasoning_text,
            re.IGNORECASE | re.DOTALL
        )
        
        if thinking_match:
            text = thinking_match.group(1).strip()
        else:
            text = reasoning_text.strip()
        
        # Extract paragraphs (split by double newlines or single newline after period)
        paragraphs = self._extract_paragraphs(text)
        
        # Parse each paragraph to extract nodes and edges
        nodes_dict: Dict[str, Node] = {}  # node_id -> Node
        edges: List[Edge] = []
        seen_edges: Set[tuple] = set()  # (source, target) tuples to avoid duplicates
        
        for paragraph in paragraphs:
            # Extract target node and value from (NODE : True/False)
            target_info = self._extract_target_node(paragraph)
            if not target_info:
                continue
            
            target_name, target_value = target_info
            target_id = target_name.upper()  # Store as uppercase
            
            # Extract source nodes from [NODE]
            source_names = self._extract_source_nodes(paragraph)
            
            # Create or update target node
            if target_id not in nodes_dict:
                nodes_dict[target_id] = Node(
                    id=target_id,
                    label=target_id,
                    value=self._parse_bool_value(target_value)
                )
            else:
                # Update value if not already set
                if nodes_dict[target_id].value is None:
                    nodes_dict[target_id].value = self._parse_bool_value(target_value)
            
            # Create edges from sources to target
            for source_name in source_names:
                source_id = source_name.upper()  # Store as uppercase
                
                # Skip self-connections
                if source_id == target_id:
                    continue
                
                # Create source node if it doesn't exist
                if source_id not in nodes_dict:
                    nodes_dict[source_id] = Node(
                        id=source_id,
                        label=source_id,
                        value=None  # Source nodes may not have values set yet
                    )
                
                # Create edge if not already seen
                edge_tuple = (source_id, target_id)
                if edge_tuple not in seen_edges:
                    edges.append(Edge(source_id, target_id))
                    seen_edges.add(edge_tuple)
        
        # Convert nodes dict to list
        nodes = list(nodes_dict.values())
        
        return Graph(nodes, edges)
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """
        Extract paragraphs from reasoning text.
        Paragraphs are separated by double newlines or single newline after period.
        """
        # Split by double newlines first
        paragraphs = re.split(r'\n\s*\n', text)
        
        # If no double newlines, try splitting by single newline after period
        if len(paragraphs) == 1:
            # Split by newline after period, question mark, or exclamation
            paragraphs = re.split(r'[.!?]\s*\n', text)
            # Re-add the punctuation to each paragraph (except the last)
            for i in range(len(paragraphs) - 1):
                paragraphs[i] = paragraphs[i] + '.'
        
        # Clean up paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def _extract_target_node(self, paragraph: str) -> Optional[tuple]:
        """
        Extract target node and value from (NODE : True/False) pattern.
        
        Returns:
            Tuple of (node_name, value_string) or None
        """
        # Pattern to match (NODE : True/False) or (NODE: True/False)
        # Case-insensitive matching
        pattern = r'\(([A-Z0-9\-_]+)\s*:\s*(True|False)\)'
        match = re.search(pattern, paragraph, re.IGNORECASE)
        
        if match:
            node_name = match.group(1)
            value_str = match.group(2)
            return (node_name, value_str)
        
        return None
    
    def _extract_source_nodes(self, paragraph: str) -> List[str]:
        """
        Extract all source nodes from [NODE] patterns.
        Case-insensitive matching.
        """
        # Pattern to match [NODE] - case insensitive
        pattern = r'\[([A-Z0-9\-_]+)\]'
        matches = re.findall(pattern, paragraph, re.IGNORECASE)
        return matches
    
    def _parse_bool_value(self, value_str: str) -> Optional[bool]:
        """
        Parse boolean value from string.
        """
        value_lower = value_str.lower().strip()
        if value_lower == 'true':
            return True
        elif value_lower == 'false':
            return False
        return None

