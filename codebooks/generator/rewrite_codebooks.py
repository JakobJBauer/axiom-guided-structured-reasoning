import os
import sys
import asyncio
from typing import List, Optional, Callable, Any
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


class CodebookRewriter:
    
    STYLES = [
        "free-flow",
        "transcript",
        "technical",
        "structured",
        "flowery",
        "concise",
        "narrative"
    ]
    
    STYLE_DESCRIPTIONS = {
        "free-flow": "Natural, conversational, flowing text that reads smoothly without rigid structure. Use the most natural language possible.",
        "transcript": "Dialogue-like, interview style with questions and answers, as if explaining to someone. Focus on questions that might arise, and answer them.",
        "technical": "Precise, formal technical language with clear definitions and specifications. Feel free to use technical jargon and acronyms.",
        "structured": "Clear, organized format with bullet points, sections, and hierarchical organization. This should make the annotator think in a structured way.",
        "flowery": "Extended, descriptive, elaborate language with rich vocabulary and detailed explanations. Make up a story behind that codebook and be creative.",
        "concise": "Brief, to-the-point style with minimal words while maintaining clarity. Do not impose too strict of a structure, but keep yourself short and concise.",
        "narrative": "Story-like, engaging narrative style that weaves concepts together like a story. It should be exciting to read and bring over the point of the codebook."
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and OPENAI_API_KEY environment variable not set")
        
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def rewrite_codebook(self, codebook_text: str, style: str) -> str:
        if style not in self.STYLES:
            raise ValueError(f"Unknown style: {style}. Must be one of {self.STYLES}")
        
        prompt = self._create_rewrite_prompt(codebook_text, style)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at rewriting technical documentation and codebooks "
                                  "in different writing styles while maintaining accuracy and logical structure."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                # temperature=0.7,  # Some creativity for style variation
            )
            
            rewritten_text = response.choices[0].message.content.strip()
            return rewritten_text
            
        except Exception as e:
            raise RuntimeError(f"Failed to rewrite codebook: {e}")
    
    async def rewrite_codebooks_parallel(
        self,
        codebook_texts: List[str],
        styles: List[str],
        max_concurrent: int = 10,
        on_complete: Optional[Callable[[int, Any], None]] = None
    ) -> List[str]:
        """
        Rewrite multiple codebooks in parallel.
        
        Args:
            codebook_texts: List of codebook texts to rewrite
            styles: List of styles (must match length of codebook_texts)
            max_concurrent: Maximum number of concurrent API calls
            on_complete: Optional callback function(index, result) called immediately when each result is ready
        
        Returns:
            List of rewritten codebook texts (in same order as inputs)
        """
        if len(codebook_texts) != len(styles):
            raise ValueError("codebook_texts and styles must have the same length")
        
        tasks = []
        for codebook_text, style in zip(codebook_texts, styles):
            if style not in self.STYLES:
                raise ValueError(f"Unknown style: {style}. Must be one of {self.STYLES}")
            
            prompt = self._create_rewrite_prompt(codebook_text, style)
            task = create_chat_task(user_message=prompt)
            tasks.append(task)
        
        results = await parallel_api_calls(
            tasks=tasks,
            api_key=self.api_key,
            model=self.model,
            max_concurrent=max_concurrent,
            system_message="You are an expert at rewriting technical documentation and codebooks "
                          "in different writing styles while maintaining accuracy and logical structure.",
            progress_desc="Rewriting codebooks",
            on_complete=on_complete
        )
        
        # Check for errors
        rewritten_texts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise RuntimeError(f"Failed to rewrite codebook {i}: {result}")
            rewritten_texts.append(result)
        
        return rewritten_texts
    
    def _create_rewrite_prompt(self, codebook_text: str, style: str) -> str:
        style_description = self.STYLE_DESCRIPTIONS[style]
        
        return f"""Rewrite the following codebook in a {style} style.

Style: {style}
Description: {style_description}

IMPORTANT REQUIREMENTS:
1. Ensure that the pragmatic logical structure of the content is still the same. So relationships between nodes should not be changed.2. Keep all node IDs in [BRACKET] format exactly as they appear
3. Preserve all logical operations (Not, And, Or, etc.) and their relationships
4. The codebook will be used by people for annotating and reasoning, so accuracy is critical
5. Improve the naturalness and readability of the text while keeping it accurate
6. Make the text more engaging and easier to understand in the {style} style. Please be creative. Feel free to restructure the content, as long as the pragmatics are preserved.
7. Do NOT change any node IDs, logical relationships, or formula structures. The pragmatics have to be the same.
8. Do NOT add or remove nodes

Original codebook:
{codebook_text}

Rewrite the codebook in {style} style, maintaining all logical structure. Be creative and really get into the role of {style}:"""
    
    def rewrite_codebook_file(
        self,
        codebook_path: str,
        style: str,
        output_path: Optional[str] = None
    ) -> str:
        with open(codebook_path, 'r', encoding='utf-8') as f:
            codebook_text = f.read()
        
        rewritten_text = self.rewrite_codebook(codebook_text, style)
        
        if output_path is None:
            original_path = Path(codebook_path)
            # Remove existing style suffixes if present
            stem = original_path.stem
            for existing_style in self.STYLES:
                if stem.endswith(f"-{existing_style}"):
                    stem = stem[:-len(f"-{existing_style}")]
            output_path = original_path.parent / f"{stem}-{style}{original_path.suffix}"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rewritten_text)
        
        return str(output_path)
    
    def rewrite_all_codebooks_in_directory(
        self,
        directory: str,
        styles: Optional[List[str]] = None,
        pattern: str = "*.txt",
        exclude_patterns: Optional[List[str]] = None
    ):
        if styles is None:
            styles = self.STYLES
        
        directory_path = Path(directory)
        codebook_files = list(directory_path.glob(pattern))
        
        if exclude_patterns:
            filtered_files = []
            for file in codebook_files:
                should_exclude = any(
                    exclude_pattern in file.name for exclude_pattern in exclude_patterns
                )
                if not should_exclude:
                    filtered_files.append(file)
            codebook_files = filtered_files
        
        original_files = []
        for file in codebook_files:
            has_style_suffix = any(
                file.stem.endswith(f"-{style}") for style in self.STYLES
            )
            if not has_style_suffix:
                original_files.append(file)
        
        total_rewrites = len(original_files) * len(styles)
        
        print(f"Found {len(original_files)} codebook files")
        print(f"Will create {total_rewrites} rewritten versions ({len(styles)} styles each)")
        print(f"Styles: {', '.join(styles)}\n")
        
        with tqdm(total=total_rewrites, desc="Rewriting codebooks") as pbar:
            for codebook_file in original_files:
                for style in styles:
                    try:
                        output_path = self.rewrite_codebook_file(
                            str(codebook_file),
                            style
                        )
                        pbar.set_postfix({
                            'file': codebook_file.name[:30],
                            'style': style
                        })
                    except Exception as e:
                        print(f"\nError rewriting {codebook_file.name} in {style} style: {e}")
                    finally:
                        pbar.update(1)
        
        print(f"\n✓ Rewriting complete!")
        print(f"  Original files: {len(original_files)}")
        print(f"  Rewritten versions: {total_rewrites}")
