import asyncio
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import aiohttp

REPO_ROOT = Path(__file__).resolve().parents[3]  # OCR-Coder/
MANIFEST_PATH = REPO_ROOT / "MVV" / "Phase_3" / "full_data" / "manifest_out.jsonl"
OUTPUT_PATH = REPO_ROOT / "MVV" / "Phase_3" / "Phase_3_4" / "reasoning_dataset.jsonl"

API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o-mini"
MAX_CONCURRENT = 50
MAX_RETRIES = 5
TEMPERATURE = 0.7
BATCH_SIZE = 20  # process 20 files concurrently


def load_api_key() -> str:
    """Load OpenAI API key from .env file, then fall back to environment variable."""
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key == "OPENAI_API_KEY":
                    os.environ["OPENAI_API_KEY"] = value
                    break

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("FATAL: OPENAI_API_KEY not found in .env file or environment.", file=sys.stderr)
        print("Create MVV/Phase_3/Phase_3_4/.env with: OPENAI_API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)
    return api_key


def load_and_group_manifest(path: Path) -> Dict[str, List[dict]]:
    """Load manifest, group by file_id, sort chunks by chunk_index."""
    groups = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = rec["file_id"]
            if fid not in groups:
                groups[fid] = []
            groups[fid].append(rec)
    # Sort each group by chunk_index
    for fid in groups:
        groups[fid].sort(key=lambda r: r["chunk_index"])
    return groups


def load_processed_ids(path: Path) -> Set[str]:
    """Read existing output to find already-processed file_ids."""
    processed = set()
    if not path.exists():
        return processed
    with open(path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                # Extract file_id from the first tensor path
                if rec.get("sequence_tensors"):
                    # tensor path looks like: MVV/Phase_3/full_data/tensors_and_texts/black__action__main_py_chunk0.pt
                    # file_id is everything before _chunk\d+.pt
                    tp = rec["sequence_tensors"][0]
                    fname = Path(tp).stem  # black__action__main_py_chunk0
                    # Find the file_id by removing _chunk\d+ suffix
                    match = re.match(r"(.+)_chunk\d+$", fname)
                    if match:
                        processed.add(match.group(1))
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    return processed


async def api_call(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_key: str,
    messages: list,
    response_format: dict = None,
) -> dict:
    """Make an OpenAI API call with semaphore throttling and exponential backoff."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
    }
    if response_format:
        payload["response_format"] = response_format

    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                async with session.post(API_URL, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    elif resp.status in (429, 502, 503):
                        wait = 2 ** attempt
                        print(f"  API {resp.status}, retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                        await asyncio.sleep(wait)
                    else:
                        body = await resp.text()
                        print(f"  API error {resp.status}: {body[:200]}")
                        await asyncio.sleep(2 ** attempt)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                wait = 2 ** attempt
                print(f"  Connection error: {e}, retrying in {wait}s")
                await asyncio.sleep(wait)

    print(f"  FAILED after {MAX_RETRIES} retries")
    return None


async def macro_pass(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_key: str,
    full_text: str,
    file_id: str,
) -> dict:
    """Generate a high-level summary Q&A for the entire file."""
    messages = [
        {
            "role": "system",
            "content": "You are an expert software architect. Read this entire Python file and write a 1-2 sentence high-level summary of its primary purpose and functionality."
        },
        {
            "role": "user",
            "content": full_text
        }
    ]
    summary = await api_call(session, semaphore, api_key, messages)
    if summary is None:
        return None
    return {
        "question": "What is the overall architecture and primary purpose of this file?",
        "answer": summary
    }


async def micro_pass(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_key: str,
    block_texts: List[str],
    block_size: int,
) -> list:
    """Generate spatial and cross-boundary Q&A for a chunk block."""
    combined_text = "\n".join(block_texts)

    if block_size == 1:
        system_prompt = (
            "You are an expert Python analyst. I will provide a code block physically "
            "contained in 1 chunk. Generate exactly one Question/Answer pair.\n\n"
            "Question (Spatial): Identify the visual scope, indentation, or nesting depth "
            "of a specific block.\n\n"
            "You must return a JSON object with a single root key called \"qa_pairs\". "
            "The value of \"qa_pairs\" must be an array containing exactly 1 object with "
            "keys \"question\" and \"answer\"."
        )
        expected_count = 1
    else:
        system_prompt = (
            f"You are an expert Python analyst. I will provide a continuous code block "
            f"physically split into {block_size} chunks. Generate exactly two "
            f"Question/Answer pairs based ONLY on this text. Assume the reader is "
            f"looking at images of these chunks.\n\n"
            f"Question 1 (Spatial): Identify the visual scope, indentation, or nesting "
            f"depth of a specific block.\n\n"
            f"Question 2 (Cross-Boundary): Trace a variable, class, or logic flow "
            f"specifically from one chunk into the next.\n\n"
            f"You must return a JSON object with a single root key called \"qa_pairs\". "
            f"The value of \"qa_pairs\" must be an array containing exactly 2 objects, "
            f"each with keys \"question\" and \"answer\"."
        )
        expected_count = 2

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": combined_text}
    ]

    response = await api_call(
        session, semaphore, api_key, messages,
        response_format={"type": "json_object"}
    )

    if response is None:
        return []

    try:
        parsed = json.loads(response)
        qa_pairs = parsed.get("qa_pairs", [])
        if not isinstance(qa_pairs, list):
            print(f"  Warning: qa_pairs is not a list, got {type(qa_pairs)}")
            return []
        return qa_pairs[:expected_count]
    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse micro response: {e}")
        return []


def append_output(record: dict, path: Path) -> None:
    """Append a single JSON record with file locking."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(line)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


async def process_file(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_key: str,
    file_id: str,
    chunks: List[dict],
) -> int:
    """Process one file: macro pass + micro passes for all 3-chunk blocks. Returns count of records written."""
    # Concatenate all chunk texts for the macro pass
    chunk_texts = [c["ground_truth_text"] for c in chunks]
    full_text = "\n".join(chunk_texts)

    # Pass 1: Macro summary
    macro_qa = await macro_pass(session, semaphore, api_key, full_text, file_id)
    if macro_qa is None:
        print(f"  Skipping {file_id}: macro pass failed")
        return 0

    # Pass 2: Micro Q&A for each 3-chunk block
    records_written = 0
    for block_start in range(0, len(chunks), 3):
        block = chunks[block_start : block_start + 3]
        block_size = len(block)
        block_texts = [c["ground_truth_text"] for c in block]

        micro_qas = await micro_pass(session, semaphore, api_key, block_texts, block_size)

        # Assemble output record
        sequence_tensors = [c["tensor_path"] for c in block]
        qa_pairs = [macro_qa] + micro_qas

        record = {
            "file_id": file_id,
            "block_start_chunk": block[0]["chunk_index"],
            "block_end_chunk": block[-1]["chunk_index"],
            "sequence_tensors": sequence_tensors,
            "qa_pairs": qa_pairs,
        }

        append_output(record, OUTPUT_PATH)
        records_written += 1

    return records_written


async def main():
    api_key = load_api_key()

    print(f"Loading manifest from {MANIFEST_PATH}")
    groups = load_and_group_manifest(MANIFEST_PATH)
    print(f"Found {len(groups)} unique files")

    # Resume logic
    processed = load_processed_ids(OUTPUT_PATH)
    remaining = {fid: chunks for fid, chunks in groups.items() if fid not in processed}
    print(f"Already processed: {len(processed)}, remaining: {len(remaining)}")

    if not remaining:
        print("All files already processed. Nothing to do.")
        return

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    total_records = 0
    total_files = 0
    start_time = time.time()

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        file_ids = sorted(remaining.keys())

        for batch_start in range(0, len(file_ids), BATCH_SIZE):
            batch = file_ids[batch_start : batch_start + BATCH_SIZE]

            tasks = [
                process_file(session, semaphore, api_key, fid, remaining[fid])
                for fid in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for fid, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"  Error processing {fid}: {result}")
                else:
                    total_records += result
                    total_files += 1

            if (batch_start + len(batch)) % 100 <= BATCH_SIZE:
                elapsed = time.time() - start_time
                rate = total_files / elapsed * 3600 if elapsed > 0 else 0
                print(f"[{batch_start + len(batch)}/{len(file_ids)}] "
                      f"{total_files} files, {total_records} records, "
                      f"{rate:.0f} files/hr, elapsed {elapsed/60:.1f}min")

    elapsed = time.time() - start_time
    print(f"\nDone! {total_files} files -> {total_records} records in {elapsed/60:.1f} minutes")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
