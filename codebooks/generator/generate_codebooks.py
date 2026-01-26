"""
Generate codebooks automatically using OpenAI API.

Creates codebooks with logical trees of varying sizes and difficulties.
Also generates obfuscated versions with renamed nodes.
"""

import os
import json
import sys
import asyncio
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import openai
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import api_utils - handle both relative and absolute imports
try:
    from .api_utils import parallel_api_calls, create_chat_task
except ImportError:
    # Fallback for direct imports (e.g., in notebooks)
    from api_utils import parallel_api_calls, create_chat_task

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
        size: str = "medium",  # "small", "medium", "large", "insane"
        difficulty: str = "medium",  # "easy", "medium", "hard"
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
    
    async def generate_codebooks_parallel(
        self,
        generation_configs: List[Dict[str, Any]],
        max_concurrent: int = 10,
        on_complete: Optional[Callable[[int, Any], None]] = None
    ) -> List[str]:
        """
        Generate multiple codebooks in parallel.
        
        Args:
            generation_configs: List of dicts with keys: leaf_nodes, size, difficulty, use_all_formulas
            max_concurrent: Maximum number of concurrent API calls
            on_complete: Optional callback function(index, result) called immediately when each result is ready
        
        Returns:
            List of generated codebook texts (in same order as configs)
        """
        tasks = []
        for config in generation_configs:
            prompt = self._create_generation_prompt(
                config["leaf_nodes"],
                config["constraints"],
                config["difficulty"],
                config.get("use_all_formulas", False)
            )
            task = create_chat_task(user_message=prompt)
            tasks.append(task)
        
        results = await parallel_api_calls(
            tasks=tasks,
            api_key=self.api_key,
            model=self.model,
            max_concurrent=max_concurrent,
            system_message="You are an expert at creating logical codebooks that define concepts through boolean logic. "
                          "You create clear, well-structured codebooks with nodes and their logical relationships.",
            progress_desc="Generating codebooks",
            on_complete=on_complete
        )
        
        # Check for errors
        codebooks = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise RuntimeError(f"Failed to generate codebook {i}: {result}")
            codebooks.append(result)
        
        return codebooks
    
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
    
    def save_codebook(self, codebook_text: str, filename: str, output_dir: str = ".", logging: bool = False):
        output_path = Path(output_dir) / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(codebook_text)
        if logging: print(f"Saved: {output_path}")
    
    def generate_all_codebooks(
        self,
        output_dir: str = "generated",
        small_count: int = 20,
        medium_count: int = 20,
        large_count: int = 10,
        insane_count: int = 5,
        max_concurrent: int = 20,
        logging: bool = False
    ):
        leaf_nodes = self.load_leaf_nodes()
        
        output_path = Path(__file__).parent / output_dir
        output_path.mkdir(exist_ok=True)
        
        size_constraints = {
            "small": {"min_nodes": 3, "max_nodes": 7, "max_depth": 3},
            "medium": {"min_nodes": 8, "max_nodes": 12, "max_depth": 4},
            "large": {"min_nodes": 13, "max_nodes": 25, "max_depth": 6},
            "insane": {"min_nodes": 26, "max_nodes": 50, "max_depth": 9}
        }
        
        codebook_num = 1
        skipped_originals = 0
        skipped_obfuscated = 0
        
        # Collect all codebooks to generate
        generation_configs = []
        codebook_metadata = []  # Store metadata for each codebook
        
        # Small codebooks
        for i in range(small_count):
            difficulty = "easy" if i < 10 else "medium"
            use_all_formulas = (i % 5 == 0)
            formula_suffix = "-allf" if use_all_formulas else ""
            filename = f"cb-{codebook_num:03d}-small-{difficulty}{formula_suffix}.txt"
            filepath = output_path / filename
            
            if not filepath.exists():
                import random
                constraints = size_constraints["small"]
                num_leaf_nodes = min(
                    random.randint(constraints["min_nodes"] - 2, constraints["max_nodes"] - 1),
                    len(leaf_nodes)
                )
                selected_leaf_nodes = random.sample(leaf_nodes, num_leaf_nodes)
                
                generation_configs.append({
                    "leaf_nodes": selected_leaf_nodes,
                    "constraints": constraints,
                    "difficulty": difficulty,
                    "use_all_formulas": use_all_formulas
                })
                codebook_metadata.append({
                    "filename": filename,
                    "filepath": filepath,
                    "codebook_num": codebook_num,
                    "size": "small",
                    "difficulty": difficulty,
                    "formula_suffix": formula_suffix
                })
            else:
                skipped_originals += 1
            
            codebook_num += 1
        
        # Medium codebooks
        for i in range(medium_count):
            difficulty = "medium" if i < 15 else "hard"
            use_all_formulas = (i % 4 == 0)
            formula_suffix = "-allf" if use_all_formulas else ""
            filename = f"cb-{codebook_num:03d}-medium-{difficulty}{formula_suffix}.txt"
            filepath = output_path / filename
            
            if not filepath.exists():
                import random
                constraints = size_constraints["medium"]
                num_leaf_nodes = min(
                    random.randint(constraints["min_nodes"] - 2, constraints["max_nodes"] - 1),
                    len(leaf_nodes)
                )
                selected_leaf_nodes = random.sample(leaf_nodes, num_leaf_nodes)
                
                generation_configs.append({
                    "leaf_nodes": selected_leaf_nodes,
                    "constraints": constraints,
                    "difficulty": difficulty,
                    "use_all_formulas": use_all_formulas
                })
                codebook_metadata.append({
                    "filename": filename,
                    "filepath": filepath,
                    "codebook_num": codebook_num,
                    "size": "medium",
                    "difficulty": difficulty,
                    "formula_suffix": formula_suffix
                })
            else:
                skipped_originals += 1
            
            codebook_num += 1
        
        # Large codebooks
        for i in range(large_count):
            difficulty = "hard"
            use_all_formulas = (i % 5 == 0)
            formula_suffix = "-allf" if use_all_formulas else ""
            filename = f"cb-{codebook_num:03d}-large-{difficulty}{formula_suffix}.txt"
            filepath = output_path / filename
            
            if not filepath.exists():
                import random
                constraints = size_constraints["large"]
                num_leaf_nodes = min(
                    random.randint(constraints["min_nodes"] - 2, constraints["max_nodes"] - 1),
                    len(leaf_nodes)
                )
                selected_leaf_nodes = random.sample(leaf_nodes, num_leaf_nodes)
                
                generation_configs.append({
                    "leaf_nodes": selected_leaf_nodes,
                    "constraints": constraints,
                    "difficulty": difficulty,
                    "use_all_formulas": use_all_formulas
                })
                codebook_metadata.append({
                    "filename": filename,
                    "filepath": filepath,
                    "codebook_num": codebook_num,
                    "size": "large",
                    "difficulty": difficulty,
                    "formula_suffix": formula_suffix
                })
            else:
                skipped_originals += 1
            
            codebook_num += 1
        
        # Insane codebooks
        for i in range(insane_count):
            difficulty = "hard"
            use_all_formulas = (i % 5 == 0)
            formula_suffix = "-allf" if use_all_formulas else ""
            filename = f"cb-{codebook_num:03d}-insane-{difficulty}{formula_suffix}.txt"
            filepath = output_path / filename
            
            if not filepath.exists():
                import random
                constraints = size_constraints["insane"]
                num_leaf_nodes = min(
                    random.randint(constraints["min_nodes"] - 2, constraints["max_nodes"] - 1),
                    len(leaf_nodes)
                )
                selected_leaf_nodes = random.sample(leaf_nodes, num_leaf_nodes)
                
                generation_configs.append({
                    "leaf_nodes": selected_leaf_nodes,
                    "constraints": constraints,
                    "difficulty": difficulty,
                    "use_all_formulas": use_all_formulas
                })
                codebook_metadata.append({
                    "filename": filename,
                    "filepath": filepath,
                    "codebook_num": codebook_num,
                    "size": "insane",
                    "difficulty": difficulty,
                    "formula_suffix": formula_suffix
                })
            else:
                skipped_originals += 1
            
            codebook_num += 1
        
        # Generate all codebooks in parallel
        if generation_configs:
            print(f"Generating {len(generation_configs)} codebooks in parallel...")
            
            # Define callback to save immediately when each codebook is generated
            def save_callback(index: int, result: Any):
                """Save codebook immediately when API call completes."""
                if isinstance(result, Exception):
                    return  # Skip errors, they'll be handled later
                
                metadata = codebook_metadata[index]
                # Save original codebook immediately
                self.save_codebook(result, metadata["filename"], output_dir=str(output_path), logging=logging)
                
                # Generate and save obfuscated version immediately
                obf_filename = f"cb-{metadata['codebook_num']:03d}-{metadata['size']}-{metadata['difficulty']}{metadata['formula_suffix']}-obfc.txt"
                obf_filepath = output_path / obf_filename
                if not obf_filepath.exists():
                    obfuscated = self.obfuscate_codebook(result)
                    self.save_codebook(obfuscated, obf_filename, output_dir=str(output_path), logging=logging)
            
            try:
                from .api_utils import run_async
            except ImportError:
                from api_utils import run_async
            codebooks = run_async(
                self.generate_codebooks_parallel(
                    generation_configs, 
                    max_concurrent=max_concurrent,
                    on_complete=save_callback
                )
            )
            
            # Verify all were saved (they should be already, but check for any errors)
            for i, (codebook_text, metadata) in enumerate(zip(codebooks, codebook_metadata)):
                filepath = output_path / metadata["filename"]
                if not filepath.exists():
                    # Re-save if somehow missed
                    self.save_codebook(codebook_text, metadata["filename"], output_dir=str(output_path), logging=logging)
        
        # Handle obfuscation for existing files
        codebook_num = 1
        for size, count in [("small", small_count), ("medium", medium_count), ("large", large_count), ("insane", insane_count)]:
            for i in range(count):
                if size == "small":
                    difficulty = "easy" if i < 10 else "medium"
                    use_all_formulas = (i % 5 == 0)
                elif size == "medium":
                    difficulty = "medium" if i < 15 else "hard"
                    use_all_formulas = (i % 4 == 0)
                else:
                    difficulty = "hard"
                    use_all_formulas = (i % 5 == 0)
                
                formula_suffix = "-allf" if use_all_formulas else ""
                filename = f"cb-{codebook_num:03d}-{size}-{difficulty}{formula_suffix}.txt"
                filepath = output_path / filename
                obf_filename = f"cb-{codebook_num:03d}-{size}-{difficulty}{formula_suffix}-obfc.txt"
                obf_filepath = output_path / obf_filename
                
                if filepath.exists() and not obf_filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as f:
                        codebook = f.read()
                    obfuscated = self.obfuscate_codebook(codebook)
                    self.save_codebook(obfuscated, obf_filename, output_dir=str(output_path), logging=logging)
                elif obf_filepath.exists():
                    skipped_obfuscated += 1
                
                codebook_num += 1

        total_generated = codebook_num - 1
        print(f"\n✓ Processed {total_generated} codebooks")
        if skipped_originals > 0:
            print(f"  Skipped {skipped_originals} existing original codebooks")
        if skipped_obfuscated > 0:
            print(f"  Skipped {skipped_obfuscated} existing obfuscated codebooks")
        print(f"  Output directory: {output_path}")
