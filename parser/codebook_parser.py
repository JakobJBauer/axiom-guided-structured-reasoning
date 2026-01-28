import os
import json
import sys
import asyncio
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import openai
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../codebooks/generator'))

# Import api_utils - handle both relative and absolute imports
try:
    from codebooks.generator.api_utils import parallel_api_calls, create_chat_task
except ImportError:
    # Fallback for direct imports or when package structure is different
    try:
        from .api_utils import parallel_api_calls, create_chat_task
    except ImportError:
        # Last resort: use importlib with explicit reload
        import importlib.util
        import importlib
        api_utils_path = os.path.join(os.path.dirname(__file__), '../codebooks/generator/api_utils.py')
        # Use a unique module name to avoid caching issues
        module_name = f"api_utils_parser_{id(api_utils_path)}"
        spec = importlib.util.spec_from_file_location(module_name, api_utils_path)
        api_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api_utils)
        parallel_api_calls = api_utils.parallel_api_calls
        create_chat_task = api_utils.create_chat_task

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

        # Default output path for the serialized graph (JSON)
        if output_path is None:
            output_path = self._get_output_path(codebook_path)
        
        graph_data = self._extract_graph_structure(codebook_text)
        graph = self._create_graph_from_data(graph_data)
        
        # Save graph using the central JSON serializer
        save_graph(graph, output_path)
        print(f"Graph saved to {output_path}")
        
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
                temperature=1.0,  # Explicitly set to 1.0 (model default)
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            graph_data = json.loads(result_text)
            
            return graph_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract graph structure from LLM: {e}")
    
    async def parse_codebooks_parallel(
        self,
        codebook_texts: List[str],
        max_concurrent: int = 10,
        on_complete: Optional[Callable[[int, Any], None]] = None
    ) -> tuple[List[Optional[Graph]], List[Optional[Dict[str, Any]]], List[Optional[str]]]:
        """
        Parse multiple codebooks in parallel.

        Args:
            codebook_texts: List of codebook texts to parse
            max_concurrent: Maximum number of concurrent API calls
            on_complete: Optional callback function(index, result) called
                         immediately when each raw LLM result is ready

        Returns:
            Tuple of:
                - list of Graph objects or None (for failed parses)
                - list of graph_data dicts or None (for failed parses)
                - list of error messages (str) or None (for successful parses)
        """
        tasks = []
        for codebook_text in codebook_texts:
            prompt = self._create_extraction_prompt(codebook_text)
            task = create_chat_task(
                user_message=prompt,
                temperature=1.0,  # Explicitly set to 1.0 (model default)
                response_format={"type": "json_object"}
            )
            tasks.append(task)

        results = await parallel_api_calls(
            tasks=tasks,
            api_key=self.api_key,
            model=self.model,
            max_concurrent=max_concurrent,
            system_message="You are an expert at analyzing codebooks and extracting logical graph structures. "
                          "You extract nodes, edges, and logical formulas from natural language descriptions.",
            progress_desc="Parsing codebooks",
            on_complete=on_complete
        )

        graphs: List[Optional[Graph]] = []
        graph_data_list: List[Optional[Dict[str, Any]]] = []
        errors: List[Optional[str]] = []

        for result in results:
            # API-level failure
            if isinstance(result, Exception):
                graphs.append(None)
                graph_data_list.append(None)
                errors.append(str(result))
                continue

            # JSON / graph construction failure
            try:
                graph_data = json.loads(result)
                graph = self._create_graph_from_data(graph_data)
                graphs.append(graph)
                graph_data_list.append(graph_data)
                errors.append(None)
            except Exception as e:
                graphs.append(None)
                graph_data_list.append(None)
                errors.append(str(e))

        failed_count = sum(1 for e in errors if e is not None)
        if failed_count:
            print(f"Warning: {failed_count} codebook(s) failed to parse; see parse_errors_log.txt for details.")

        return graphs, graph_data_list, errors
    
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
      // IMPORTANT: For nested formulas, use nested objects:
      // "formula_args": ["node1", {{"formula_type": "Or", "formula_args": ["node2", "node3"]}}]
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
7. CRITICAL: For nested formulas (e.g., "X and (Y or Z)"), represent them as nested JSON objects:
   - Do NOT use string representations like "And(X, Or(Y, Z))"
   - Instead use: ["X", {{"formula_type": "Or", "formula_args": ["Y", "Z"]}}]
   - Example: "A is B and (C or D)" should be:
     {{"formula_type": "And", "formula_args": ["b", {{"formula_type": "Or", "formula_args": ["c", "d"]}}]}}

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
    
    def _create_formula(self, formula_type: str, args: List[Any]) -> Any:
        """
        Create a Formula object from type and arguments.
        
        Args:
            formula_type: Type of formula (Not, And, Or, etc.)
            args: Arguments for the formula (node IDs should be lowercase, or nested formula dicts)
        
        Returns:
            Formula object
        """
        # Process arguments: handle nested formulas and normalize node IDs
        processed_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, dict) and "formula_type" in arg:
                # This is a nested formula - recursively create it
                nested_formula = self._create_formula(
                    arg["formula_type"],
                    arg.get("formula_args", [])
                )
                processed_args.append(nested_formula)
            elif isinstance(arg, str) and formula_type not in ["Equal", "In"]:
                # Normalize node IDs to lowercase
                processed_args.append(arg.lower())
            elif isinstance(arg, str) and formula_type in ["Equal", "In"]:
                # For Equal/In, first arg is node ID (normalize), rest are values (don't normalize)
                if i == 0:
                    processed_args.append(arg.lower())
                else:
                    processed_args.append(arg)
            else:
                processed_args.append(arg)
        
        if formula_type == "Not":
            if len(processed_args) != 1:
                raise ValueError(f"Not formula requires 1 argument, got {len(processed_args)}")
            return Not(processed_args[0])
        
        elif formula_type == "And":
            if len(processed_args) < 1:
                raise ValueError(f"And formula requires at least 1 argument, got {len(processed_args)}")
            return And(*processed_args)
        
        elif formula_type == "Or":
            if len(processed_args) < 1:
                raise ValueError(f"Or formula requires at least 1 argument, got {len(processed_args)}")
            return Or(*processed_args)
        
        elif formula_type == "Xor":
            if len(processed_args) < 1:
                raise ValueError(f"Xor formula requires at least 1 argument, got {len(processed_args)}")
            return Xor(*processed_args)
        
        elif formula_type == "Equal":
            if len(processed_args) != 2:
                raise ValueError(f"Equal formula requires 2 arguments, got {len(processed_args)}")
            # First arg is node ID (normalized), second is value (parse)
            value = self._parse_value(processed_args[1])
            return Equal(processed_args[0], value)
        
        elif formula_type == "In":
            if len(processed_args) < 2:
                raise ValueError(f"In formula requires at least 2 arguments, got {len(processed_args)}")
            key = processed_args[0]  # Node ID (normalized)
            if isinstance(processed_args[1], list):
                values = self._parse_value(processed_args[1])
            else:
                values = [self._parse_value(v) for v in processed_args[1:]]
            return In(key, values)
        
        else:
            raise ValueError(f"Unknown formula type: {formula_type}")
    
    def _parse_value(self, value: Any) -> Any:
        """
        Robustly parse a value that may be a string, list, or primitive.
        
        - Lists: recursively parse each element
        - Strings: strip and try bool/int/float, else keep as string
        - Other types: return as-is (already a primitive or structured value)
        """
        # Handle lists: parse each element
        if isinstance(value, list):
            return [self._parse_value(v) for v in value]
        
        # Non-string, non-list: return as-is (already a primitive or structured value)
        if not isinstance(value, str):
            return value
        
        value_str = value.strip()
        
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
        return str(path.with_suffix('.json'))
