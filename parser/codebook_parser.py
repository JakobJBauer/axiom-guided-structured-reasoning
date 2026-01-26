import os
import json
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path
import openai
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph import Node, Edge, Graph
from graph.formulas import Not, And, Or, Xor, Equal, In
from serializer import save_graph

load_dotenv()


class CodebookParser:
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and OPENAI_API_KEY environment variable not set")
        
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def parse_codebook(self, codebook_path: str, output_path: Optional[str] = None) -> Graph:
        with open(codebook_path, 'r', encoding='utf-8') as f:
            codebook_text = f.read()

        if output_path is None: output_path = self._get_output_path(codebook_path)
        
        graph_data = self._extract_graph_structure(codebook_text)
        graph = self._create_graph_from_data(graph_data)
        
        # Save pickle file
        save_graph(graph, output_path)
        print(f"Graph saved to {output_path}")
        
        # Save JSON file (same name/location, different extension)
        json_path = self._get_json_output_path(output_path)
        self._save_graph_json(graph_data, json_path)
        print(f"Graph JSON saved to {json_path}")
        
        return graph
    
    def _extract_graph_structure(self, codebook_text: str) -> Dict[str, Any]:
        prompt = self._create_extraction_prompt(codebook_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing codebooks and extracting logical graph structures. "
                                  "You extract nodes, edges, and logical formulas from natural language descriptions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,  # Use deterministic output
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            graph_data = json.loads(result_text)
            
            return graph_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract graph structure from LLM: {e}")
    
    def _create_extraction_prompt(self, codebook_text: str) -> str:
        return f"""Analyze the following codebook and extract the graph structure.

A codebook defines nodes (concepts) and their logical relationships. Each node can have:
- A formula that defines it based on other nodes (using logical operations)
- Edges from prerequisite nodes to the defined node

Logical operations:
- "not X" or "is not X" -> Not("X")
- "both X and Y" or "X and Y" -> And("X", "Y")
- "either X or Y" or "X or Y" -> Or("X", "Y")
- "X equals Y" -> Equal("X", Y)
- "X in [list]" -> In("X", [list])

For nodes without formulas (leaf nodes), they are defined by external conditions.

Extract the following structure and return as JSON:

{{
  "nodes": [
    {{
      "id": "node_id",  // lowercase, use hyphens for multi-word
      "label": "Node Label",  // human-readable label
      "formula_type": "Not|And|Or|Xor|Equal|In|null",  // null for leaf nodes
      "formula_args": ["arg1", "arg2"]  // arguments for the formula, or [] for leaf nodes
    }}
  ],
  "edges": [
    {{
      "source": "source_node_id",
      "target": "target_node_id"
    }}
  ]
}}

Rules:
1. Create edges from all nodes mentioned in a formula to the node being defined
2. For "not X", create edge from X to the node
3. For "X and Y", create edges from both X and Y to the node
4. For "X or Y", create edges from both X and Y to the node
5. Node IDs should be lowercase with hyphens (e.g., "non-noun", "well-selling")
6. Only include nodes that are explicitly defined in the codebook

Codebook:
{codebook_text}

Return only valid JSON, no additional text."""
    
    def _create_graph_from_data(self, graph_data: Dict[str, Any]) -> Graph:
        nodes = []
        edges = []
        
        # Create nodes
        node_id_to_node = {}
        for node_data in graph_data.get("nodes", []):
            node_id = node_data["id"].lower()
            label = node_data.get("label", node_id)
            
            # Create formula if specified
            formula = None
            formula_type = node_data.get("formula_type")
            formula_args = node_data.get("formula_args", [])
            
            if formula_type:
                formula = self._create_formula(formula_type, formula_args)
            
            node = Node(
                id=node_id,
                label=label,
                formula=formula
            )
            nodes.append(node)
            node_id_to_node[node_id] = node
        
        # Create edges
        for edge_data in graph_data.get("edges", []):
            source = edge_data["source"].lower()
            target = edge_data["target"].lower()
            
            # Verify nodes exist
            if source not in node_id_to_node or target not in node_id_to_node:
                print(f"Warning: Edge {source} -> {target} references non-existent node, skipping")
                continue
            
            edges.append(Edge(source, target))
        
        return Graph(nodes, edges)
    
    def _create_formula(self, formula_type: str, args: List[str]) -> Any:
        """
        Create a Formula object from type and arguments.
        
        Args:
            formula_type: Type of formula (Not, And, Or, etc.)
            args: Arguments for the formula (node IDs should be lowercase)
        
        Returns:
            Formula object
        """
        # Normalize node IDs to lowercase (for all string args except values in Equal/In)
        normalized_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, str) and formula_type not in ["Equal", "In"]:
                normalized_args.append(arg.lower())
            elif isinstance(arg, str) and formula_type in ["Equal", "In"]:
                # For Equal/In, first arg is node ID (normalize), rest are values (don't normalize)
                if i == 0:
                    normalized_args.append(arg.lower())
                else:
                    normalized_args.append(arg)
            else:
                normalized_args.append(arg)
        
        if formula_type == "Not":
            if len(normalized_args) != 1:
                raise ValueError(f"Not formula requires 1 argument, got {len(normalized_args)}")
            return Not(normalized_args[0])
        
        elif formula_type == "And":
            if len(normalized_args) < 1:
                raise ValueError(f"And formula requires at least 1 argument, got {len(normalized_args)}")
            return And(*normalized_args)
        
        elif formula_type == "Or":
            if len(normalized_args) < 1:
                raise ValueError(f"Or formula requires at least 1 argument, got {len(normalized_args)}")
            return Or(*normalized_args)
        
        elif formula_type == "Xor":
            if len(normalized_args) < 1:
                raise ValueError(f"Xor formula requires at least 1 argument, got {len(normalized_args)}")
            return Xor(*normalized_args)
        
        elif formula_type == "Equal":
            if len(normalized_args) != 2:
                raise ValueError(f"Equal formula requires 2 arguments, got {len(normalized_args)}")
            # First arg is node ID (normalized), second is value (parse)
            value = self._parse_value(normalized_args[1])
            return Equal(normalized_args[0], value)
        
        elif formula_type == "In":
            if len(normalized_args) < 2:
                raise ValueError(f"In formula requires at least 2 arguments, got {len(normalized_args)}")
            key = normalized_args[0]  # Node ID (normalized)
            values = [self._parse_value(v) for v in normalized_args[1:]]
            return In(key, values)
        
        else:
            raise ValueError(f"Unknown formula type: {formula_type}")
    
    def _parse_value(self, value_str: str) -> Any:
        value_str = value_str.strip()
        
        # Try boolean
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False
        
        # Try integer
        try:
            return int(value_str)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value_str)
        except ValueError:
            pass
        
        # Return as string
        return value_str
    
    def _get_output_path(self, codebook_path: str) -> str:
        path = Path(codebook_path)
        return str(path.with_suffix('.pkl'))
    
    def _get_json_output_path(self, pickle_path: str) -> str:
        path = Path(pickle_path)
        return str(path.with_suffix('.json'))
    
    def _save_graph_json(self, graph_data: Dict[str, Any], json_path: str) -> None:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
