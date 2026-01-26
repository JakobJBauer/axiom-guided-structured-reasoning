"""
Complete codebook generation pipeline.

This script runs the entire pipeline:
1. Generate codebooks
2. Obfuscate original codebooks
3. Rewrite codebooks in different styles
4. Obfuscate rewritten codebooks
5. Parse all codebooks into graphs
6. Serialize all graphs (pickle + JSON)

All files are saved in the same directory with appropriate suffixes.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from generate_codebooks import CodebookGenerator
from rewrite_codebooks import CodebookRewriter
from parser import CodebookParser

load_dotenv()


class CodebookPipeline:    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        rewrite_styles: Optional[List[str]] = None
    ):
        self.generator = CodebookGenerator(api_key=api_key, model=model)
        self.rewriter = CodebookRewriter(api_key=api_key, model=model)
        self.parser = CodebookParser(api_key=api_key, model=model)
        self.rewrite_styles = rewrite_styles if rewrite_styles is not None else CodebookRewriter.STYLES
    
    def run_full_pipeline(
        self,
        output_dir: str,
        small_count: int = 20,
        medium_count: int = 15,
        large_count: int = 10,
        insane_count: int = 5
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("CODEBOOK GENERATION PIPELINE")
        print("=" * 80)
        print(f"Output directory: {output_path.absolute()}")
        print(f"Rewriting styles: {', '.join(self.rewrite_styles)}\n")
        
        print("Step 1: Generating codebooks...")
        print("-" * 80)
        self._generate_codebooks(
            output_path,
            small_count,
            medium_count,
            large_count,
            insane_count
        )
        
        print("\nStep 2: Obfuscating original codebooks...")
        print("-" * 80)
        self._obfuscate_codebooks(output_path, exclude_patterns=["-obfc", "-flowery", "-technical", "-free-flow", "-transcript", "-structured", "-concise", "-narrative"])

        print("\nStep 3: Rewriting codebooks in different styles...")
        print("-" * 80)
        self._rewrite_codebooks(output_path)
        
        print("\nStep 4: Obfuscating rewritten codebooks...")
        print("-" * 80)
        self._obfuscate_rewritten_codebooks(output_path)
        
        print("\nStep 5: Parsing and serializing all codebooks...")
        print("-" * 80)
        self._parse_and_serialize_all(output_path)
        
        print("\n" + "=" * 80)
        print("✓ PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"All files saved to: {output_path.absolute()}")
    
    def _generate_codebooks(
        self,
        output_path: Path,
        small_count: int,
        medium_count: int,
        large_count: int,
        insane_count: int
    ):
        self.generator.generate_all_codebooks(
            output_dir=str(output_path),
            small_count=small_count,
            medium_count=medium_count,
            large_count=large_count,
            insane_count=insane_count
        )
    
    def _obfuscate_codebooks(self, output_path: Path, exclude_patterns: List[str]):
        codebook_files = list(output_path.glob("*.txt"))
        
        # Filter: only original codebooks (no -obfc, no style suffixes)
        original_files = []
        style_suffixes = [f"-{style}" for style in CodebookRewriter.STYLES]
        for file in codebook_files:
            has_obfc = "-obfc" in file.stem
            has_style = any(file.stem.endswith(suffix) for suffix in style_suffixes)
            should_exclude = any(pattern in file.stem for pattern in exclude_patterns)
            
            if not has_obfc and not has_style and not should_exclude:
                original_files.append(file)
        
        if not original_files:
            print("No original codebooks to obfuscate.")
            return
        
        print(f"Obfuscating {len(original_files)} original codebooks...")
        
        with tqdm(total=len(original_files), desc="Obfuscating") as pbar:
            for codebook_file in original_files:
                try:
                    with open(codebook_file, 'r', encoding='utf-8') as f:
                        codebook_text = f.read()
                    
                    obfuscated = self.generator.obfuscate_codebook(codebook_text)
                    
                    obfuscated_file = codebook_file.parent / f"{codebook_file.stem}-obfc{codebook_file.suffix}"
                    with open(obfuscated_file, 'w', encoding='utf-8') as f:
                        f.write(obfuscated)
                    
                    pbar.set_postfix({'file': codebook_file.name[:30]})
                except Exception as e:
                    print(f"\nError obfuscating {codebook_file.name}: {e}")
                finally:
                    pbar.update(1)
    
    def _rewrite_codebooks(self, output_path: Path):
        codebook_files = list(output_path.glob("*.txt"))
        
        original_files = []
        style_suffixes = [f"-{style}" for style in CodebookRewriter.STYLES]
        for file in codebook_files:
            has_style = any(file.stem.endswith(suffix) for suffix in style_suffixes)
            has_obfc = "-obfc" in file.stem
            if not has_style and not has_obfc:
                original_files.append(file)
        
        if not original_files:
            print("No original codebooks to rewrite.")
            return
        
        total_rewrites = len(original_files) * len(self.rewrite_styles)
        print(f"Rewriting {len(original_files)} codebooks in {len(self.rewrite_styles)} styles...")
        
        with tqdm(total=total_rewrites, desc="Rewriting") as pbar:
            for codebook_file in original_files:
                for style in self.rewrite_styles:
                    try:
                        rewritten_path = self.rewriter.rewrite_codebook_file(
                            str(codebook_file),
                            style
                        )
                        pbar.set_postfix({
                            'file': codebook_file.name[:25],
                            'style': style
                        })
                    except Exception as e:
                        print(f"\nError rewriting {codebook_file.name} in {style}: {e}")
                    finally:
                        pbar.update(1)
    
    def _obfuscate_rewritten_codebooks(self, output_path: Path):
        codebook_files = list(output_path.glob("*.txt"))
        
        # Find rewritten codebooks (have style suffix but no -obfc)
        rewritten_files = []
        style_suffixes = [f"-{style}" for style in CodebookRewriter.STYLES]
        for file in codebook_files:
            has_style = any(file.stem.endswith(suffix) for suffix in style_suffixes)
            has_obfc = "-obfc" in file.stem
            if has_style and not has_obfc:
                rewritten_files.append(file)
        
        if not rewritten_files:
            print("No rewritten codebooks to obfuscate.")
            return
        
        print(f"Obfuscating {len(rewritten_files)} rewritten codebooks...")
        
        with tqdm(total=len(rewritten_files), desc="Obfuscating rewritten") as pbar:
            for codebook_file in rewritten_files:
                try:
                    # Read rewritten
                    with open(codebook_file, 'r', encoding='utf-8') as f:
                        codebook_text = f.read()
                    
                    # Obfuscate
                    obfuscated = self.generator.obfuscate_codebook(codebook_text)
                    
                    # Save with -obfc suffix (insert before .txt)
                    obfuscated_file = codebook_file.parent / f"{codebook_file.stem}-obfc{codebook_file.suffix}"
                    with open(obfuscated_file, 'w', encoding='utf-8') as f:
                        f.write(obfuscated)
                    
                    pbar.set_postfix({'file': codebook_file.name[:30]})
                except Exception as e:
                    print(f"\nError obfuscating {codebook_file.name}: {e}")
                finally:
                    pbar.update(1)
    
    def _parse_and_serialize_all(self, output_path: Path):
        """Parse all codebook files and serialize them as graphs."""
        codebook_files = list(output_path.glob("*.txt"))
        
        if not codebook_files:
            print("No codebook files to parse.")
            return
        
        print(f"Parsing and serializing {len(codebook_files)} codebook files...")
        
        successful = 0
        failed = 0
        
        with tqdm(total=len(codebook_files), desc="Parsing & serializing") as pbar:
            for codebook_file in codebook_files:
                try:
                    # Parse codebook (this automatically saves both .pkl and .json)
                    # Temporarily suppress parser's print statements
                    import io
                    import contextlib
                    
                    f = io.StringIO()
                    with contextlib.redirect_stdout(f):
                        graph = self.parser.parse_codebook(str(codebook_file))
                    
                    successful += 1
                    pbar.set_postfix({
                        'file': codebook_file.name[:25],
                        'success': successful,
                        'failed': failed
                    })
                except Exception as e:
                    failed += 1
                    print(f"\nError parsing {codebook_file.name}: {e}")
                finally:
                    pbar.update(1)
        
        print(f"\nParsing complete: {successful} successful, {failed} failed")


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete codebook generation and processing pipeline"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for all generated files"
    )
    parser.add_argument("--small", type=int, default=20, help="Number of small codebooks")
    parser.add_argument("--medium", type=int, default=20, help="Number of medium codebooks")
    parser.add_argument("--large", type=int, default=10, help="Number of large codebooks")
    parser.add_argument("--insane", type=int, default=5, help="Number of insane codebooks")
    parser.add_argument(
        "--styles",
        nargs="+",
        choices=CodebookRewriter.STYLES,
        help="Styles for rewriting (default: all styles)"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--api-key", help="API key (default: reads from OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    try:
        pipeline = CodebookPipeline(
            api_key=args.api_key,
            model=args.model,
            rewrite_styles=args.styles
        )
        
        pipeline.run_full_pipeline(
            output_dir=args.output_dir,
            small_count=args.small,
            medium_count=args.medium,
            large_count=args.large,
            insane_count=args.insane
        )
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

