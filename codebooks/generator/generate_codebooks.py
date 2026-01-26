"""
Generate codebooks automatically using OpenAI API.

Creates codebooks with logical trees of varying sizes and difficulties.
Also generates obfuscated versions with renamed nodes.
"""

import os
import json
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import openai
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

load_dotenv()


class CodebookGenerator:
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and OPENAI_API_KEY environment variable not set")
        
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def load_leaf_nodes(self, leaf_nodes_file: str = "proposed_leaf_nodes.json") -> List[Dict[str, Any]]:
        leaf_nodes_path = Path(__file__).parent / leaf_nodes_file
        with open(leaf_nodes_path, 'r') as f:
            return json.load(f)
    
    def generate_codebook(
        self,
        leaf_nodes: List[Dict[str, Any]],
        size: str = "medium",  # "small", "medium", "large"
        difficulty: str = "medium",  # "easy", "medium", "hard", "insane"
        use_all_formulas: bool = False
    ) -> str:
        size_constraints = {
            "small": {"min_nodes": 3, "max_nodes": 7, "max_depth": 3},
            "medium": {"min_nodes": 8, "max_nodes": 12, "max_depth": 4},
            "large": {"min_nodes": 13, "max_nodes": 25, "max_depth": 6},
            "insane": {"min_nodes": 26, "max_nodes": 50, "max_depth": 9}
        }
        
        constraints = size_constraints[size]
        
        # Select leaf nodes for this codebook
        import random
        num_leaf_nodes = min(
            random.randint(constraints["min_nodes"] - 2, constraints["max_nodes"] - 1),
            len(leaf_nodes)
        )
        selected_leaf_nodes = random.sample(leaf_nodes, num_leaf_nodes)
        
        # Build prompt
        prompt = self._create_generation_prompt(
            selected_leaf_nodes,
            constraints,
            difficulty,
            use_all_formulas
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating logical codebooks that define concepts through boolean logic. "
                                  "You create clear, well-structured codebooks with nodes and their logical relationships."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                # temperature=0.7,  # Some creativity
            )
            
            codebook_text = response.choices[0].message.content.strip()
            return codebook_text
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate codebook: {e}")
    
    def _create_generation_prompt(
        self,
        leaf_nodes: List[Dict[str, Any]],
        constraints: Dict[str, int],
        difficulty: str,
        use_all_formulas: bool
    ) -> str:
        """Create prompt for codebook generation."""
        
        leaf_descriptions = "\n".join([
            f"[{node['id'].upper().replace('-', '-')}]\n{node['description']}"
            for node in leaf_nodes
        ])
        
        formula_requirements = ""
        if use_all_formulas:
            formula_requirements = """
You MUST use all of the following logical operations at least once:
- Not (negation)
- And (conjunction - both conditions must be true)
- Or (disjunction - either condition can be true)
- Xor (exclusive or - exactly one condition must be true)
- Equal (equality check)
- In (membership check)
"""
        else:
            formula_requirements = """
You can use any combination of these logical operations:
- Not (negation): "is not X"
- And (conjunction): "both X and Y are true"
- Or (disjunction): "either X or Y is true"
- Xor (exclusive or): "exactly one of X or Y is true"
- Equal (equality): "X equals value"
- In (membership): "X is in [list]" [list] is a comma-separated list of values to check membership against.
"""
        
        difficulty_guidance = {
            "easy": "Keep formulas simple. Use mostly And and Or operations. Avoid deep nesting.",
            "medium": "Use moderate complexity. Include some Not operations and 2-3 level nesting.",
            "hard": "Use complex formulas with deep nesting (3-4 levels), multiple formula types, and intricate logical relationships."
        }
        
        return f"""Create a codebook that defines logical relationships between concepts.

AVAILABLE LEAF NODES (these are the base concepts you can use). You need to define each leaf node that you use within the codebook:
{leaf_descriptions}

CONSTRAINTS:
- Total nodes (including leaf nodes): {constraints['min_nodes']} to {constraints['max_nodes']}
- Maximum depth: {constraints['max_depth']} levels
- Difficulty: {difficulty}
  {difficulty_guidance[difficulty]}

{formula_requirements}

FORMAT - FOLLOW THIS EXACTLY:
Each node definition must follow this format:

[NODE-ID]
A story is [NODE-ID] if [description/logical definition].

EXAMPLES:

[SHORT]
A story is short if it contains fewer than 150 words.

[NON-NOUN]
A story is non-noun if it is not [NOUN].

[DENSE]
A story is dense if both of the following are true:
- The story is [NON-NOUN]
- The story is [SHORT]

[THRILLING]
A story is thrilling if either of the following is true:
- The story is [MAGICAL]
- The story is [SERIOUS]

RULES:
1. Node IDs in brackets must be UPPERCASE with hyphens: [NODE-ID]
2. First line after bracket: "A story is [NODE-ID] if"
3. Reference other nodes using [NODE-ID] format
4. Leave a blank line between node definitions
5. For leaf nodes, use the description provided
6. Create meaningful intermediate nodes that combine concepts logically
7. Any node that is mentioned, must be defined in the codebook. Also the nodes that you receive as available leaf nodes, must be defined in the codebook.

Generate the codebook now, following this format exactly:"""
    
    def obfuscate_codebook(self, codebook_text: str) -> str:
        import re
        
        # Extract all node IDs
        node_pattern = r'\[([A-Z0-9\-_]+)\]'
        nodes = set(re.findall(node_pattern, codebook_text, re.IGNORECASE))
        
        # Create mapping: original -> obfuscated
        node_mapping = {}
        attr_counter = 1
        
        # Sort nodes for consistent mapping
        sorted_nodes = sorted(nodes, key=str.lower)
        
        for node in sorted_nodes:
            node_mapping[node] = f"attr-{attr_counter}"
            attr_counter += 1
        
        # Replace all occurrences
        obfuscated_text = codebook_text
        for original, obfuscated in node_mapping.items():
            # Replace in brackets (case-insensitive)
            obfuscated_text = re.sub(
                rf'\[{re.escape(original)}\]',
                f'[{obfuscated}]',
                obfuscated_text,
                flags=re.IGNORECASE
            )
        
        return obfuscated_text
    
    def save_codebook(self, codebook_text: str, filename: str, output_dir: str = "."):
        output_path = Path(output_dir) / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(codebook_text)
        print(f"Saved: {output_path}")
    
    def generate_all_codebooks(
        self,
        output_dir: str = "generated",
        small_count: int = 20,
        medium_count: int = 20,
        large_count: int = 10,
        insane_count: int = 5
    ):
        leaf_nodes = self.load_leaf_nodes()
        
        output_path = Path(__file__).parent / output_dir
        output_path.mkdir(exist_ok=True)
        
        codebook_num = 1
        
        # Generate small codebooks
        for i in tqdm(range(small_count), desc="Small codebooks"):
            difficulty = "easy" if i < 10 else "medium"
            use_all_formulas = (i % 5 == 0)  # Every 5th uses all formulas
            
            codebook = self.generate_codebook(
                leaf_nodes,
                size="small",
                difficulty=difficulty,
                use_all_formulas=use_all_formulas
            )
            
            # Create filename with size and difficulty
            formula_suffix = "-allf" if use_all_formulas else ""
            filename = f"cb-{codebook_num:03d}-small-{difficulty}{formula_suffix}.txt"
            self.save_codebook(codebook, filename, output_dir=str(output_path))
            
            # Generate obfuscated version
            obfuscated = self.obfuscate_codebook(codebook)
            obf_filename = f"cb-{codebook_num:03d}-small-{difficulty}{formula_suffix}-obfc.txt"
            self.save_codebook(obfuscated, obf_filename, output_dir=str(output_path))
            
            codebook_num += 1
        
        # Generate medium codebooks
        for i in tqdm(range(medium_count), desc="Medium codebooks"):
            difficulty = "medium" if i < 15 else "hard"
            use_all_formulas = (i % 4 == 0)  # Every 4th uses all formulas
            
            codebook = self.generate_codebook(
                leaf_nodes,
                size="medium",
                difficulty=difficulty,
                use_all_formulas=use_all_formulas
            )
            
            # Create filename with size and difficulty
            formula_suffix = "-allf" if use_all_formulas else ""
            filename = f"cb-{codebook_num:03d}-medium-{difficulty}{formula_suffix}.txt"
            self.save_codebook(codebook, filename, output_dir=str(output_path))
            
            # Generate obfuscated version
            obfuscated = self.obfuscate_codebook(codebook)
            obf_filename = f"cb-{codebook_num:03d}-medium-{difficulty}{formula_suffix}-obfc.txt"
            self.save_codebook(obfuscated, obf_filename, output_dir=str(output_path))
            
            codebook_num += 1
        
        # Generate large codebooks
        for i in tqdm(range(large_count), desc="Large codebooks"):
            difficulty = "hard"
            use_all_formulas = (i % 5 == 0)
            
            codebook = self.generate_codebook(
                leaf_nodes,
                size="large",
                difficulty=difficulty,
                use_all_formulas=use_all_formulas
            )
            
            # Create filename with size and difficulty
            formula_suffix = "-allf" if use_all_formulas else ""
            filename = f"cb-{codebook_num:03d}-large-{difficulty}{formula_suffix}.txt"
            self.save_codebook(codebook, filename, output_dir=str(output_path))
            
            # Generate obfuscated version
            obfuscated = self.obfuscate_codebook(codebook)
            obf_filename = f"cb-{codebook_num:03d}-large-{difficulty}{formula_suffix}-obfc.txt"
            self.save_codebook(obfuscated, obf_filename, output_dir=str(output_path))
            
            codebook_num += 1

        # Generate insane codebooks
        for i in tqdm(range(insane_count), desc="Insane codebooks"):
            difficulty = "insane"
            use_all_formulas = (i % 5 == 0)
            
            codebook = self.generate_codebook(
                leaf_nodes,
                size="insane",
                difficulty=difficulty,
                use_all_formulas=use_all_formulas
            )
            
            # Create filename with size and difficulty
            formula_suffix = "-allf" if use_all_formulas else ""
            filename = f"cb-{codebook_num:03d}-insane-{difficulty}{formula_suffix}.txt"
            self.save_codebook(codebook, filename, output_dir=str(output_path))
            
            # Generate obfuscated version
            obfuscated = self.obfuscate_codebook(codebook)
            obf_filename = f"cb-{codebook_num:03d}-insane-{difficulty}{formula_suffix}-obfc.txt"
            self.save_codebook(obfuscated, obf_filename, output_dir=str(output_path))
            
            codebook_num += 1

        print(f"\n✓ Generated {codebook_num - 1} codebooks (and obfuscated versions)")
        print(f"  Output directory: {output_path}")
