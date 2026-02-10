"""
code_to_image.py — Convert Python source files to syntax-highlighted PNG images.

Optimized for OCR extraction with DeepSeek-OCR-2.

Usage:
    # Single file
    python code_to_image.py examples/fibonacci.py

    # Multiple files
    python code_to_image.py examples/*.py

    # Custom output directory and style
    python code_to_image.py examples/fibonacci.py -o ./code_images --style monokai

    # Try all styles (for comparing OCR accuracy)
    python code_to_image.py examples/fibonacci.py --all-styles

    # Batch a whole directory
    python code_to_image.py --dir path/to/repo -o ./code_images
"""

import argparse
import os
import sys
from pathlib import Path

from pygments import highlight
from pygments.lexers import PythonLexer, get_lexer_for_filename
from pygments.formatters import ImageFormatter
from pygments.styles import get_all_styles
from PIL import Image

# Fonts known to exist on Rosie / standard Linux
FONT_CANDIDATES = [
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Noto Sans Mono",
    "Courier New",
    "Courier",
]

# Styles worth testing for OCR readability (light backgrounds first — best for OCR)
OCR_STYLES = ["default", "friendly", "monokai", "github-dark"]


def find_working_font() -> str:
    """Return the first monospace font name Pygments can actually use."""
    test_code = "x = 1"
    for font in FONT_CANDIDATES:
        try:
            fmt = ImageFormatter(font_name=font, font_size=14)
            highlight(test_code, PythonLexer(), fmt)
            return font
        except Exception:
            continue
    return "Courier"  # ultimate fallback built into Pillow


def convert_code_to_image(
    code_file_path: str,
    output_dir: str = "./code_images",
    style: str = "default",
    font_size: int = 14,
    line_numbers: bool = True,
    image_pad: int = 10,
) -> str:
    """
    Convert a source code file to a syntax-highlighted PNG image.

    Args:
        code_file_path: Path to the source file.
        output_dir:     Directory to write the PNG into.
        style:          Pygments style name (e.g. "default", "monokai").
        font_size:      Font size in points.
        line_numbers:   Whether to render line numbers on the left.
        image_pad:      Pixel padding around the code block.

    Returns:
        Absolute path to the generated PNG.
    """
    code_path = Path(code_file_path)
    if not code_path.is_file():
        raise FileNotFoundError(f"Source file not found: {code_file_path}")

    code = code_path.read_text()

    # Pick lexer based on file extension (falls back to Python)
    try:
        lexer = get_lexer_for_filename(code_path.name)
    except Exception:
        lexer = PythonLexer()

    font_name = find_working_font()

    formatter = ImageFormatter(
        style=style,
        font_name=font_name,
        font_size=font_size,
        line_numbers=line_numbers,
        image_pad=image_pad,
        line_number_bg="#f0f0f0",
        line_number_fg="#888888",
    )

    image_bytes = highlight(code, lexer, formatter)

    # Write to output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = code_path.stem
    out_name = f"{stem}_{style}.png"
    out_path = out_dir / out_name
    out_path.write_bytes(image_bytes)

    return str(out_path.resolve())


def convert_all_styles(
    code_file_path: str,
    output_dir: str = "./code_images",
    font_size: int = 14,
    line_numbers: bool = True,
) -> list[str]:
    """Generate an image for every style in OCR_STYLES. Returns list of paths."""
    paths = []
    for style in OCR_STYLES:
        try:
            p = convert_code_to_image(
                code_file_path,
                output_dir=output_dir,
                style=style,
                font_size=font_size,
                line_numbers=line_numbers,
            )
            paths.append(p)
        except Exception as e:
            print(f"  [WARN] Style '{style}' failed: {e}")
    return paths


def image_info(image_path: str) -> dict:
    """Return width, height, and file size for a generated image."""
    p = Path(image_path)
    img = Image.open(p)
    return {
        "path": str(p),
        "width": img.width,
        "height": img.height,
        "file_size_kb": round(p.stat().st_size / 1024, 1),
    }


def count_text_tokens(code: str) -> int:
    """Rough token estimate for raw text (GPT-style ~4 chars/token)."""
    return len(code) // 4


def print_report(code_file: str, image_paths: list[str]):
    """Print a summary table of generated images and token estimates."""
    code = Path(code_file).read_text()
    text_tokens = count_text_tokens(code)
    lines = code.count("\n") + 1

    print()
    print("=" * 70)
    print(f"  Source: {code_file}")
    print(f"  Lines:  {lines}    Characters: {len(code)}    Est. text tokens: {text_tokens}")
    print("-" * 70)
    print(f"  {'Style':<16} {'Size (KB)':>10} {'Width':>7} {'Height':>8}  Path")
    print("-" * 70)

    for ip in image_paths:
        info = image_info(ip)
        style = Path(ip).stem.split("_")[-1]
        print(
            f"  {style:<16} {info['file_size_kb']:>10} {info['width']:>7} {info['height']:>8}  {info['path']}"
        )

    print("=" * 70)
    print()


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert Python source files to syntax-highlighted PNG images for OCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python code_to_image.py examples/fibonacci.py
  python code_to_image.py examples/*.py --all-styles
  python code_to_image.py --dir examples/ -o code_images
        """,
    )
    parser.add_argument(
        "files", nargs="*", help="Source file(s) to convert"
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Convert every .py file in this directory",
    )
    parser.add_argument(
        "-o", "--output", default="./code_images",
        help="Output directory for images (default: ./code_images)",
    )
    parser.add_argument(
        "--style", default="default",
        help="Pygments style (default: default). Use --all-styles to try several.",
    )
    parser.add_argument(
        "--all-styles", action="store_true",
        help="Generate images for all OCR-candidate styles",
    )
    parser.add_argument(
        "--font-size", type=int, default=14,
        help="Font size in pt (default: 14)",
    )
    parser.add_argument(
        "--no-line-numbers", action="store_true",
        help="Disable line numbers",
    )

    args = parser.parse_args()

    # Collect input files
    files: list[str] = list(args.files or [])
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"[ERROR] Not a directory: {args.dir}")
            sys.exit(1)
        files.extend(str(f) for f in sorted(dir_path.glob("*.py")))

    if not files:
        parser.print_help()
        sys.exit(0)

    line_numbers = not args.no_line_numbers

    print(f"Converting {len(files)} file(s) -> {args.output}/")
    print(f"Style: {'all OCR candidates' if args.all_styles else args.style}")
    print(f"Font size: {args.font_size}pt   Line numbers: {line_numbers}")

    for code_file in files:
        if args.all_styles:
            paths = convert_all_styles(
                code_file,
                output_dir=args.output,
                font_size=args.font_size,
                line_numbers=line_numbers,
            )
        else:
            paths = [
                convert_code_to_image(
                    code_file,
                    output_dir=args.output,
                    style=args.style,
                    font_size=args.font_size,
                    line_numbers=line_numbers,
                )
            ]
        print_report(code_file, paths)

    print("Done.")


if __name__ == "__main__":
    main()
