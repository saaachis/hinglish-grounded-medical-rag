"""Extract structured clinical evidence from MultiCaRe case narratives.

Uses an LLM (Ollama local or Google Gemini API) to transform long clinical
case narratives into concise structured evidence suitable for RAG grounding.

Features:
- Priority-based processing (highest MMCQSD-impact conditions first)
- Proportional sampling (3x MMCQSD query count per condition)
- Incremental saving (resumable if interrupted)
- Multiple LLM backend support (Ollama, Gemini)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

FILTERED_INPUT = Path("data/processed/multicare_filtered.csv")
EVIDENCE_OUTPUT = Path("data/processed/multicare_evidence.csv")

# Priority ordering: highest MMCQSD query count first
CONDITION_PRIORITY: list[tuple[str, int]] = [
    ("skin_rash", 1050),
    ("neck_swelling", 276),
    ("mouth_ulcers", 196),
    ("lip_swelling", 193),
    ("swollen_tonsils", 174),
    ("foot_swelling", 172),
    ("hand_lump", 162),
    ("swollen_eye", 152),
    ("knee_swelling", 115),
    ("edema", 112),
    ("eye_redness", 92),
    ("skin_growth", 81),
    ("skin_irritation", 77),
    ("skin_dryness", 65),
    ("dry_scalp", 44),
    ("eye_inflammation", 25),
    ("cyanosis", 15),
    ("itchy_eyelid", 14),
]

SAMPLE_MULTIPLIER = 3
MAX_CASE_WORDS = 1500
MIN_CASE_WORDS = 50

EXTRACTION_PROMPT = """You are a clinical evidence extractor. Given a medical case report, extract ONLY the factual clinical findings into a structured format. Do NOT add any information not present in the case. Be concise.

Case Report:
{case_text}

Extract in this exact format (use "not specified" if information is absent):
Primary Finding: [main diagnosis or presenting condition]
Location: [body area affected]
Symptoms: [patient-reported symptoms, comma-separated]
Clinical Signs: [examination findings, comma-separated]
Severity: [mild / moderate / severe / not specified]
Duration: [how long symptoms present, or "not specified"]
Key Evidence: [single most important clinical sentence from the case, quoted exactly]"""


def _truncate_case(text: str, max_words: int = MAX_CASE_WORDS) -> str:
    """Truncate case text to max_words, keeping the clinical presentation."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _parse_extraction(raw_output: str) -> dict[str, str]:
    """Parse the structured extraction from LLM output."""
    fields = {
        "primary_finding": "",
        "location": "",
        "symptoms": "",
        "clinical_signs": "",
        "severity": "",
        "duration": "",
        "key_evidence": "",
    }

    patterns = {
        "primary_finding": r"Primary Finding:\s*(.+?)(?:\n|$)",
        "location": r"Location:\s*(.+?)(?:\n|$)",
        "symptoms": r"Symptoms:\s*(.+?)(?:\n|$)",
        "clinical_signs": r"Clinical Signs:\s*(.+?)(?:\n|$)",
        "severity": r"Severity:\s*(.+?)(?:\n|$)",
        "duration": r"Duration:\s*(.+?)(?:\n|$)",
        "key_evidence": r"Key Evidence:\s*(.+?)(?:\n|$)",
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, raw_output, re.IGNORECASE)
        if match:
            fields[field] = match.group(1).strip().strip('"').strip("'")

    return fields


def _build_sample_plan(
    df: pd.DataFrame,
    multiplier: int = SAMPLE_MULTIPLIER,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a priority-ordered sample of cases for extraction."""
    sampled_parts: list[pd.DataFrame] = []

    for condition, mmcqsd_count in CONDITION_PRIORITY:
        target = mmcqsd_count * multiplier
        subset = df[df["condition_group"] == condition].copy()

        # Prefer medium-length cases
        subset = subset[
            (subset["word_count"] >= MIN_CASE_WORDS)
            & (subset["word_count"] <= MAX_CASE_WORDS)
        ]

        if len(subset) == 0:
            subset = df[df["condition_group"] == condition].copy()

        actual = min(target, len(subset))
        if actual > 0:
            sample = subset.sample(n=actual, random_state=seed)
            sample = sample.copy()
            sample["priority"] = CONDITION_PRIORITY.index(
                (condition, mmcqsd_count)
            ) + 1
            sampled_parts.append(sample)
            logger.info(
                "  [P%d] %s: sampled %d / %d available (target %d)",
                sample["priority"].iloc[0], condition,
                actual, len(subset), target,
            )

    result = pd.concat(sampled_parts, ignore_index=True)
    result = result.sort_values("priority").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# LLM Backend Abstraction
# ---------------------------------------------------------------------------

class OllamaBackend:
    """Local Ollama LLM backend."""

    def __init__(self, model: str = "llama3.1:8b"):
        import ollama as _ollama
        self._client = _ollama.Client()
        self.model = model
        self._client.list()
        logger.info("Ollama backend ready with model: %s", model)

    def generate(self, prompt: str) -> str:
        import ollama as _ollama
        response = self._client.generate(model=self.model, prompt=prompt)
        return response["response"]


class GeminiBackend:
    """Google Gemini API backend (free tier)."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self.model_name = model
        self._rpm_delay = 4.5  # ~13 RPM to stay under 15 RPM limit
        logger.info("Gemini backend ready with model: %s", model)

    def generate(self, prompt: str) -> str:
        import google.generativeai as genai
        response = self._model.generate_content(prompt)
        time.sleep(self._rpm_delay)
        return response.text


class GroqBackend:
    """Groq API backend (free tier)."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self.model = model
        logger.info("Groq backend ready with model: %s", model)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        time.sleep(1.5)  # stay under 30 RPM
        return response.choices[0].message.content


def _create_backend(
    backend_name: str,
    api_key: str | None = None,
    model: str | None = None,
):
    """Create the appropriate LLM backend."""
    if backend_name == "ollama":
        return OllamaBackend(model=model or "llama3.1:8b")
    elif backend_name == "gemini":
        if not api_key:
            raise ValueError(
                "Gemini API key required. Get one free at https://ai.google.dev\n"
                "Pass via --api-key or set GEMINI_API_KEY env var."
            )
        return GeminiBackend(api_key=api_key, model=model or "gemini-2.5-flash-lite")
    elif backend_name == "groq":
        if not api_key:
            raise ValueError(
                "Groq API key required. Get one free at https://console.groq.com\n"
                "Pass via --api-key or set GROQ_API_KEY env var."
            )
        return GroqBackend(api_key=api_key, model=model or "llama-3.3-70b-versatile")
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def load_existing_progress(output_path: Path) -> set[str]:
    """Load case IDs already processed (for resume support)."""
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_csv(output_path)
        done = set(existing["case_id"].astype(str).tolist())
        logger.info("Found %d already-extracted cases (resuming)", len(done))
        return done
    except Exception:
        return set()


def extract_evidence(
    input_path: Path,
    output_path: Path,
    backend,
    max_cases: int | None = None,
    priority_groups: list[int] | None = None,
) -> pd.DataFrame:
    """Run the evidence extraction pipeline.

    Parameters
    ----------
    input_path : Path
        Path to multicare_filtered.csv
    output_path : Path
        Path to save extracted evidence (appended incrementally)
    backend
        LLM backend instance (Ollama, Gemini, or Groq)
    max_cases : int | None
        Cap on total cases to process (for testing)
    priority_groups : list[int] | None
        Only process these priority groups (1-indexed). None = all.
    """
    df = pd.read_csv(input_path)
    logger.info("Loaded %d filtered cases from %s", len(df), input_path)

    # Build priority-ordered sample
    logger.info("Building priority-ordered sample plan...")
    sample = _build_sample_plan(df)
    logger.info("Sample plan: %d total cases across %d conditions",
                len(sample), sample["condition_group"].nunique())

    # Filter by priority groups if specified
    if priority_groups:
        sample = sample[sample["priority"].isin(priority_groups)].copy()
        logger.info("Filtered to priority groups %s: %d cases",
                    priority_groups, len(sample))

    if max_cases and len(sample) > max_cases:
        sample = sample.head(max_cases).copy()
        logger.info("Capped at %d cases", max_cases)

    # Load progress for resume
    done_ids = load_existing_progress(output_path)
    remaining = sample[~sample["case_id"].astype(str).isin(done_ids)]
    logger.info("%d cases remaining after resume check", len(remaining))

    if remaining.empty:
        logger.info("All cases already processed!")
        return pd.read_csv(output_path)

    # Process
    results: list[dict] = []
    errors = 0
    write_header = not output_path.exists()

    pbar = tqdm(
        remaining.iterrows(),
        total=len(remaining),
        desc="Extracting evidence",
    )

    for idx, row in pbar:
        case_id = str(row["case_id"])
        condition = row["condition_group"]
        case_text = _truncate_case(str(row["case_text"]))

        prompt = EXTRACTION_PROMPT.format(case_text=case_text)

        try:
            raw_output = backend.generate(prompt)
            fields = _parse_extraction(raw_output)

            record = {
                "case_id": case_id,
                "condition_group": condition,
                "primary_finding": fields["primary_finding"],
                "location": fields["location"],
                "symptoms": fields["symptoms"],
                "clinical_signs": fields["clinical_signs"],
                "severity": fields["severity"],
                "duration": fields["duration"],
                "key_evidence": fields["key_evidence"],
                "case_text": case_text[:2000],  # truncated for storage
                "extraction_quality": "pass" if fields["primary_finding"] else "failed",
                "priority": int(row["priority"]),
            }
            results.append(record)

            # Incremental save every 25 records
            if len(results) >= 25:
                batch_df = pd.DataFrame(results)
                batch_df.to_csv(
                    output_path, mode="a", header=write_header,
                    index=False, encoding="utf-8",
                )
                write_header = False
                pbar.set_postfix(saved=len(done_ids) + len(results), errors=errors)
                done_ids.update(r["case_id"] for r in results)
                results.clear()

        except KeyboardInterrupt:
            logger.info("Interrupted! Saving progress...")
            break
        except Exception as e:
            errors += 1
            logger.warning("Error on case %s: %s", case_id, str(e)[:100])
            if errors > 50:
                logger.error("Too many errors (>50). Stopping.")
                break

    # Save remaining
    if results:
        batch_df = pd.DataFrame(results)
        batch_df.to_csv(
            output_path, mode="a", header=write_header,
            index=False, encoding="utf-8",
        )
        logger.info("Saved final batch of %d records", len(batch_df))

    # Load and return full result
    if output_path.exists():
        full = pd.read_csv(output_path)
        logger.info("Total extracted: %d cases", len(full))
        return full
    return pd.DataFrame()


def print_extraction_summary(output_path: Path) -> None:
    """Print summary of extraction progress."""
    if not output_path.exists():
        print("No extraction results yet.")
        return

    df = pd.read_csv(output_path)
    passed = df[df["extraction_quality"] == "pass"]
    failed = df[df["extraction_quality"] == "failed"]

    print("\n" + "=" * 60)
    print("EXTRACTION PROGRESS SUMMARY")
    print("=" * 60)
    print(f"Total extracted:  {len(df)}")
    print(f"  Passed:         {len(passed)} ({len(passed)/len(df)*100:.1f}%)")
    print(f"  Failed:         {len(failed)} ({len(failed)/len(df)*100:.1f}%)")
    print()

    print("Per-Condition Progress:")
    print(f"{'Condition':<20} {'Done':>6} {'Target':>8} {'Pass%':>7}")
    print("-" * 45)
    for condition, mmcqsd_count in CONDITION_PRIORITY:
        target = mmcqsd_count * SAMPLE_MULTIPLIER
        done = len(df[df["condition_group"] == condition])
        pass_count = len(
            df[(df["condition_group"] == condition)
               & (df["extraction_quality"] == "pass")]
        )
        pass_pct = (pass_count / done * 100) if done > 0 else 0
        print(f"  {condition:<18} {done:>6} / {target:<6} {pass_pct:>6.1f}%")
    print()


def main() -> None:
    import os

    parser = argparse.ArgumentParser(
        description="Extract structured evidence from MultiCaRe cases"
    )
    parser.add_argument(
        "--backend", choices=["ollama", "gemini", "groq"], default="gemini",
        help="LLM backend to use (default: gemini)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key for Gemini or Groq (or set GEMINI_API_KEY / GROQ_API_KEY env var)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override default model name for the backend",
    )
    parser.add_argument(
        "--input", type=Path, default=FILTERED_INPUT,
        help="Path to filtered MultiCaRe CSV",
    )
    parser.add_argument(
        "--output", type=Path, default=EVIDENCE_OUTPUT,
        help="Path to save extracted evidence CSV",
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Cap on total cases to process (for testing)",
    )
    parser.add_argument(
        "--priority", type=int, nargs="+", default=None,
        help="Only process these priority groups (1-18). E.g.: --priority 1 2 3",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Just print extraction progress and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.status:
        print_extraction_summary(args.output)
        return

    # Resolve API key
    api_key = args.api_key
    if not api_key and args.backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and args.backend == "groq":
        api_key = os.environ.get("GROQ_API_KEY")

    # Create backend
    backend = _create_backend(args.backend, api_key=api_key, model=args.model)

    # Run extraction
    extract_evidence(
        input_path=args.input,
        output_path=args.output,
        backend=backend,
        max_cases=args.max_cases,
        priority_groups=args.priority,
    )

    print_extraction_summary(args.output)


if __name__ == "__main__":
    main()
