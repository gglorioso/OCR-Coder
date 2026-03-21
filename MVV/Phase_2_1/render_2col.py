#!/usr/bin/env python3
"""
render_2col.py — 2-column syntax-highlighted code renderer (800×800 → 448×448)

Produces 4 PNG files:
  dummy_2col_800.png  — 800×800 render of the embedded dummy snippet
  dummy_2col_448.png  — 448×448 downsampled version
  ast_2col_800.png    — 800×800 render of a real Python file (AST-extracted)
  ast_2col_448.png    — 448×448 downsampled version
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont
from pygments.lexers import PythonLexer
from pygments.styles import get_style_by_name
from pygments import token as TT

# ── Constants ─────────────────────────────────────────────────────────────────
OUT_DIR     = Path(__file__).parent
FONT_PATH   = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SIZE   = 16
CHAR_W      = 10   # px per character
LINE_H      = 20   # px per line
CANVAS_W    = 800
CANVAS_H    = 800
BG_COLOR    = "#272822"          # Monokai Dark background
GUTTER_COLOR = "#272822"         # same background (no visible gutter)
WRAP_GLYPH  = "↳"
WRAP_COLOR  = "#75715E"          # muted grey for wrap indicator

COL_LEFT_X  = 0
COL_LEFT_END = 390
GUTTER_X    = 390
GUTTER_END  = 410
COL_RIGHT_X = 410
COL_RIGHT_END = 800
MAX_CHARS   = 39    # chars per column
MAX_PHYS    = 40    # physical lines per column (40 left + 40 right = 80 total)
HANG_INDENT = 8     # extra spaces for continuation indent

# ── Real dataset file ─────────────────────────────────────────────────────────
REAL_SOURCE_FILE = (
    "/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder/"
    "Scraped Repos/black/tests/data/cases/allow_empty_first_line.py"
)

# ── Dummy snippet ─────────────────────────────────────────────────────────────
DUMMY_CODE = '''class DataPipelineOrchestrator:
    """Orchestrates the full data pipeline with retry logic."""

    def __init__(self, configuration_dictionary, maximum_retry_attempts=3):
        self.configuration_dictionary = configuration_dictionary
        self.maximum_retry_attempts = maximum_retry_attempts
        self.processed_items_accumulator = []
        self.failed_items_registry = {}

    def process_batch_with_validation(self, input_data_batch, validation_threshold=0.95):
        successfully_processed_items = []
        for item_index, current_data_item in enumerate(input_data_batch):
            if self._validate_single_item(current_data_item, validation_threshold):
                transformed_item = self._apply_transformation_pipeline(current_data_item)
                if transformed_item is not None:
                    successfully_processed_items.append(transformed_item)
                    self.processed_items_accumulator.append(transformed_item)
                else:
                    self.failed_items_registry[item_index] = current_data_item
            else:
                self.failed_items_registry[item_index] = current_data_item
        return successfully_processed_items

    def _validate_single_item(self, data_item, threshold):
        validation_score = sum(
            feature_value * weight_coefficient
            for feature_value, weight_coefficient in zip(data_item.features, self.configuration_dictionary['weights'])
        )
        return validation_score >= threshold

    def _apply_transformation_pipeline(self, data_item):
        try:
            intermediate_representation = self._encode_features(data_item.features)
            normalized_representation = [val / max(intermediate_representation) for val in intermediate_representation]
            final_representation = self._project_to_embedding_space(normalized_representation)
            return final_representation
        except (ValueError, ZeroDivisionError) as transformation_error:
            print(f"Transformation failed for item: {transformation_error}")
            return None

    def _encode_features(self, raw_feature_vector):
        return [feature ** 2 + 0.001 for feature in raw_feature_vector]

    def _project_to_embedding_space(self, normalized_vector):
        embedding_dimension = self.configuration_dictionary.get('embedding_dim', 128)
        projected = [sum(normalized_vector[i % len(normalized_vector)] for i in range(embedding_dimension))]
        return projected

    def generate_summary_report(self):
        total_processed = len(self.processed_items_accumulator)
        total_failed = len(self.failed_items_registry)
        success_rate = total_processed / (total_processed + total_failed) if (total_processed + total_failed) > 0 else 0.0
        return {
            'total_processed': total_processed,
            'total_failed': total_failed,
            'success_rate': success_rate,
            'failed_indices': list(self.failed_items_registry.keys()),
        }
'''


# ── Pygments helpers ───────────────────────────────────────────────────────────

def _build_style_map(style):
    """Return a dict mapping Pygments token types → hex color strings."""
    style_map = {}
    for token_type, style_def in style:
        if style_def.get("color"):
            style_map[token_type] = "#" + style_def["color"]
    return style_map


def _resolve_color(token_type, style_map, default="#F8F8F2"):
    """Walk up the Pygments token hierarchy to find the nearest defined color."""
    tt = token_type
    while tt is not None:
        if tt in style_map:
            return style_map[tt]
        # Move to parent token type
        if tt.parent is None or tt == tt.parent:
            break
        tt = tt.parent
    return default


def tokenize_line(line: str, style, lexer) -> List[Tuple[str, str]]:
    """
    Tokenize a single logical line and return a list of (text, color_hex) tuples.

    Uses Pygments to lex the entire line. Newline tokens are stripped.
    """
    style_map = _build_style_map(style)
    default_color = "#F8F8F2"  # Monokai foreground

    tokens = list(lexer.get_tokens(line))
    result = []
    for ttype, value in tokens:
        # Skip newline and whitespace-only token at end
        text = value.replace("\n", "")
        if not text:
            continue
        color = _resolve_color(ttype, style_map, default_color)
        result.append((text, color))
    return result


# ── Soft-wrap helper ───────────────────────────────────────────────────────────

def soft_wrap(
    logical_line: str,
    max_chars: int = MAX_CHARS,
) -> List[Tuple[str, bool, int]]:
    """
    Soft-wrap a logical line into physical lines.

    Returns a list of tuples:
        (text, is_continuation, parent_indent_spaces)

    Where:
        text                — the physical line text (without leading spaces for continuations)
        is_continuation     — True if this is a wrapped continuation line
        parent_indent_spaces — number of leading spaces of the original line
    """
    # Count leading spaces (tabs expanded to 4 spaces)
    expanded = logical_line.expandtabs(4)
    indent_count = len(expanded) - len(expanded.lstrip())

    if len(expanded) <= max_chars:
        return [(expanded, False, indent_count)]

    physical_lines = []
    remaining = expanded

    # First physical line: take up to max_chars
    first_chunk = remaining[:max_chars]
    physical_lines.append((first_chunk, False, indent_count))
    remaining = remaining[max_chars:]

    # Continuation indent = parent indent + HANG_INDENT
    cont_indent = " " * (indent_count + HANG_INDENT)
    # Available chars on continuation = max_chars - len(WRAP_GLYPH) - len(cont_indent)
    # The wrap glyph takes 1 char, then a space, so glyph_prefix is "↳ " = 2 display chars
    # We reserve 2 chars for the glyph + space
    cont_prefix_len = len(cont_indent) + 2  # "↳ "
    cont_avail = max_chars - cont_prefix_len

    while remaining:
        if cont_avail <= 0:
            # Edge case: can't fit anything meaningful; just break at max_chars
            chunk = remaining[:max(1, max_chars - len(cont_indent) - 2)]
        else:
            chunk = remaining[:cont_avail]
        physical_lines.append((chunk, True, indent_count))
        remaining = remaining[len(chunk):]

    return physical_lines


# ── Renderer ───────────────────────────────────────────────────────────────────

def render_code_to_image(
    logical_lines: List[str],
    output_path_800: Path,
    output_path_448: Path,
) -> None:
    """
    Render up to 80 physical lines across two columns and save PNG files.

    Left column:  physical lines 1–40  (x: 0–390)
    Right column: physical lines 41–80 (x: 410–800)
    """
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        print(f"WARNING: font not found at {FONT_PATH}, using default", file=sys.stderr)
        font = ImageFont.load_default()

    style = get_style_by_name("monokai")
    lexer = PythonLexer(stripnl=False, ensurenl=False)

    # ── Flatten logical lines → physical lines ────────────────────────────────
    # Each entry: (logical_line_text, is_continuation, parent_indent_spaces)
    all_physical: List[Tuple[str, bool, int]] = []
    for logical_line in logical_lines:
        wrapped = soft_wrap(logical_line, MAX_CHARS)
        all_physical.extend(wrapped)
        if len(all_physical) >= MAX_PHYS * 2:
            break

    all_physical = all_physical[: MAX_PHYS * 2]  # cap at 80

    # ── Build image ───────────────────────────────────────────────────────────
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    def _render_physical_line(phys_idx: int, line_text: str, is_cont: bool, parent_indent: int):
        """Render one physical line onto `img`."""
        # Determine column and y
        if phys_idx < MAX_PHYS:
            col_x = COL_LEFT_X
            y = phys_idx * LINE_H
        else:
            col_x = COL_RIGHT_X
            y = (phys_idx - MAX_PHYS) * LINE_H

        x = col_x

        if is_cont:
            # Draw continuation indent (spaces, same BG — just advance x)
            indent_spaces = parent_indent + HANG_INDENT
            x += indent_spaces * CHAR_W

            # Draw wrap glyph in WRAP_COLOR
            draw.text((x, y), WRAP_GLYPH, font=font, fill=WRAP_COLOR)
            x += CHAR_W  # glyph width
            # Draw a space after glyph
            x += CHAR_W

            # Now tokenize and draw the continuation content
            token_pairs = tokenize_line(line_text, style, lexer)
            for text, color in token_pairs:
                draw.text((x, y), text, font=font, fill=color)
                x += len(text) * CHAR_W
        else:
            # Normal line: tokenize the full text
            token_pairs = tokenize_line(line_text, style, lexer)
            for text, color in token_pairs:
                draw.text((x, y), text, font=font, fill=color)
                x += len(text) * CHAR_W

    for idx, (line_text, is_cont, parent_indent) in enumerate(all_physical):
        _render_physical_line(idx, line_text, is_cont, parent_indent)

    # ── Save 800×800 ─────────────────────────────────────────────────────────
    img.save(str(output_path_800), format="PNG", optimize=False)
    print(f"  Saved: {output_path_800}")

    # ── Downsample to 448×448 ─────────────────────────────────────────────────
    img_small = img.resize((448, 448), Image.Resampling.BICUBIC)
    img_small.save(str(output_path_448), format="PNG", optimize=False)
    print(f"  Saved: {output_path_448}")


# ── AST extraction ─────────────────────────────────────────────────────────────

def extract_ast_sample(source_path: str) -> List[str]:
    """
    Extract the source lines of the first function or class definition
    found in the given Python file.

    Falls back to the first 80 lines if AST parsing fails.
    """
    try:
        src = Path(source_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"WARNING: could not read {source_path}: {e}", file=sys.stderr)
        return []

    all_lines = src.splitlines()

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"WARNING: SyntaxError in {source_path}: {e}", file=sys.stderr)
        return all_lines[:80]

    target_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if target_node is None or node.lineno < target_node.lineno:
                target_node = node

    if target_node is None:
        return all_lines[:80]

    start_line = target_node.lineno - 1  # 0-indexed
    end_line = getattr(target_node, "end_lineno", start_line + 40)  # 0-indexed end

    extracted = all_lines[start_line:end_line]
    return extracted


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== render_2col.py ===")
    print(f"Output directory: {OUT_DIR}")

    # ── 1. Dummy snippet ──────────────────────────────────────────────────────
    print("\n[1/2] Rendering dummy snippet...")
    dummy_lines = DUMMY_CODE.splitlines()
    # Strip leading/trailing blank lines
    while dummy_lines and not dummy_lines[0].strip():
        dummy_lines.pop(0)
    while dummy_lines and not dummy_lines[-1].strip():
        dummy_lines.pop()

    render_code_to_image(
        logical_lines=dummy_lines,
        output_path_800=OUT_DIR / "dummy_2col_800.png",
        output_path_448=OUT_DIR / "dummy_2col_448.png",
    )

    # ── 2. AST sample from real dataset file ──────────────────────────────────
    print("\n[2/2] Rendering AST sample from real file...")
    print(f"  Source: {REAL_SOURCE_FILE}")

    ast_lines = extract_ast_sample(REAL_SOURCE_FILE)
    if not ast_lines:
        print("  WARNING: no lines extracted, using fallback empty snippet", file=sys.stderr)
        ast_lines = ["# (no source extracted)"]

    print(f"  Extracted {len(ast_lines)} logical lines")

    render_code_to_image(
        logical_lines=ast_lines,
        output_path_800=OUT_DIR / "ast_2col_800.png",
        output_path_448=OUT_DIR / "ast_2col_448.png",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Done ===")
    output_files = [
        "dummy_2col_800.png",
        "dummy_2col_448.png",
        "ast_2col_800.png",
        "ast_2col_448.png",
    ]
    for fname in output_files:
        fpath = OUT_DIR / fname
        size_kb = fpath.stat().st_size / 1024 if fpath.exists() else 0
        status = "OK" if fpath.exists() else "MISSING"
        print(f"  [{status}] {fname}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
