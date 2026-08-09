"""Download and filter MultiCaRe clinical cases for MMCQSD condition matching.

Downloads multi-specialty clinical case data from MultiCaRe (PubMed Central
open-access case reports) and filters to conditions relevant to MMCQSD's
18 medical condition categories.

The filtered output serves as the evidence corpus for the multi-specialty
RAG pipeline, replacing the chest-only Open-i approach.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

RAW_OUTPUT_DIR = Path("data/raw/multicare")
PROCESSED_OUTPUT_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# MMCQSD condition → MultiCaRe keyword mapping
#
# Each condition group defines keywords to search in MultiCaRe case text.
# A case matches if ANY keyword in the group is found in its narrative.
# ---------------------------------------------------------------------------
CONDITION_KEYWORD_MAP: dict[str, list[str]] = {
    "skin_rash": [
        "rash", "dermatitis", "eczema", "urticaria", "erythema",
        "papule", "papular", "maculopapular", "vesicular", "pustular",
        "pruritus", "pruritic", "skin lesion", "cutaneous lesion",
        "exanthem", "eruption", "skin eruption",
    ],
    "skin_growth": [
        "skin growth", "skin mass", "skin tumor", "skin nodule",
        "papilloma", "keratosis", "wart", "nevus", "mole",
        "melanocytic", "basal cell", "squamous cell",
        "cutaneous mass", "subcutaneous mass", "skin lump",
    ],
    "skin_irritation": [
        "skin irritation", "contact dermatitis", "allergic dermatitis",
        "irritant dermatitis", "prurigo", "lichenification",
        "scaling", "desquamation", "excoriation",
    ],
    "skin_dryness": [
        "dry skin", "xerosis", "ichthyosis", "xerotic",
        "skin dryness", "scaly skin", "flaky skin",
    ],
    "dry_scalp": [
        "scalp", "seborrheic dermatitis", "dandruff",
        "scalp psoriasis", "tinea capitis", "alopecia",
        "scalp lesion", "scalp rash",
    ],
    "neck_swelling": [
        "neck swelling", "neck mass", "neck lump",
        "cervical lymphadenopathy", "cervical mass",
        "thyroid nodule", "thyroid mass", "goiter", "goitre",
        "parotid", "submandibular", "lymph node neck",
    ],
    "mouth_ulcers": [
        "mouth ulcer", "oral ulcer", "aphthous", "stomatitis",
        "oral lesion", "oral mucosa", "buccal", "gingival",
        "tongue ulcer", "palatal ulcer", "oral cavity lesion",
        "mucosal ulcer",
    ],
    "lip_swelling": [
        "lip swelling", "lip edema", "lip mass", "lip lesion",
        "cheilitis", "angioedema lip", "lip enlargement",
        "labial swelling", "lower lip", "upper lip",
    ],
    "swollen_tonsils": [
        "tonsil", "tonsillar", "tonsillitis", "peritonsillar",
        "pharyngitis", "pharyngeal", "oropharyngeal",
        "throat swelling", "sore throat",
    ],
    "foot_swelling": [
        "foot swelling", "pedal edema", "ankle swelling",
        "foot mass", "foot lesion", "plantar",
        "lower extremity edema", "lower limb swelling",
        "ankle edema", "foot lump",
    ],
    "hand_lump": [
        "hand lump", "hand mass", "hand swelling", "hand lesion",
        "finger swelling", "finger mass", "wrist mass",
        "ganglion cyst", "hand nodule", "palmar",
        "dorsal hand", "metacarpal",
    ],
    "swollen_eye": [
        "eye swelling", "periorbital edema", "periorbital swelling",
        "eyelid swelling", "eyelid edema", "orbital swelling",
        "orbital mass", "proptosis", "exophthalmos",
        "lid swelling", "puffy eye",
    ],
    "knee_swelling": [
        "knee swelling", "knee effusion", "knee mass",
        "knee joint", "patellar", "prepatellar",
        "baker cyst", "knee lump", "joint effusion knee",
        "knee arthritis", "knee pain swelling",
    ],
    "edema": [
        "edema", "oedema", "anasarca", "fluid retention",
        "peripheral edema", "pitting edema", "generalized edema",
        "pulmonary edema", "cerebral edema",
        "lymphedema", "lymphoedema",
    ],
    "eye_redness": [
        "eye redness", "red eye", "conjunctivitis",
        "conjunctival injection", "subconjunctival hemorrhage",
        "episcleritis", "scleritis", "uveitis",
        "keratitis", "corneal", "ocular redness",
    ],
    "eye_inflammation": [
        "eye inflammation", "ocular inflammation",
        "iritis", "iridocyclitis", "anterior uveitis",
        "posterior uveitis", "panuveitis", "endophthalmitis",
        "orbital cellulitis", "dacryocystitis",
    ],
    "cyanosis": [
        "cyanosis", "cyanotic", "blue discoloration",
        "peripheral cyanosis", "central cyanosis",
        "acrocyanosis", "hypoxemia", "desaturation",
    ],
    "itchy_eyelid": [
        "itchy eyelid", "eyelid dermatitis", "blepharitis",
        "eyelid eczema", "eyelid pruritus",
        "eyelid irritation", "meibomian",
    ],
}

# Broader specialty-level filters applied via MultiCaRe image types
IMAGE_TYPE_CONDITION_MAP: dict[str, list[str]] = {
    "medical_photograph": [
        "skin_rash", "skin_growth", "skin_irritation", "skin_dryness",
        "dry_scalp", "mouth_ulcers", "lip_swelling", "neck_swelling",
        "foot_swelling", "hand_lump", "knee_swelling",
    ],
    "ophthalmic_imaging": [
        "swollen_eye", "eye_redness", "eye_inflammation", "itchy_eyelid",
    ],
}


def _clean_for_search(text: str) -> str:
    """Lowercase and normalize whitespace for keyword matching."""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def _assign_condition_groups(case_text: str) -> list[str]:
    """Find all MMCQSD condition groups matching a case narrative."""
    cleaned = _clean_for_search(case_text)
    matched: list[str] = []
    for condition, keywords in CONDITION_KEYWORD_MAP.items():
        if any(kw in cleaned for kw in keywords):
            matched.append(condition)
    return matched


def download_multicare_cases(output_dir: Path) -> pd.DataFrame:
    """Download MultiCaRe clinical cases using the multiversity library.

    Falls back to HuggingFace datasets library if multiversity fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "multicare_cases_raw.parquet"

    if cache_path.exists():
        logger.info("Loading cached MultiCaRe cases from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Downloading MultiCaRe dataset (this may take 5-10 minutes)...")

    # --- Attempt 1: Use multiversity library ---
    try:
        from multiversity.multicare_dataset import MedicalDatasetCreator

        data_dir = str(output_dir / "multicare_data")
        logger.info("Initializing MedicalDatasetCreator at %s ...", data_dir)
        mdc = MedicalDatasetCreator(directory=data_dir)

        cases_df = mdc.cases_df.copy()
        logger.info(
            "Loaded %d clinical cases via multiversity", len(cases_df)
        )

        cases_df.to_parquet(cache_path, index=False)
        return cases_df

    except Exception as e:
        logger.warning("multiversity download failed: %s. Trying HuggingFace...", e)

    # --- Attempt 2: Use HuggingFace datasets ---
    try:
        from datasets import load_dataset

        ds = load_dataset("openmed-community/multicare-cases", split="train")
        cases_df = ds.to_pandas()
        logger.info(
            "Loaded %d clinical cases via HuggingFace", len(cases_df)
        )

        cases_df.to_parquet(cache_path, index=False)
        return cases_df

    except Exception as e:
        logger.warning("HuggingFace download failed: %s. Trying direct parquet...", e)

    # --- Attempt 3: Direct parquet download ---
    try:
        from huggingface_hub import hf_hub_download

        parquet_path = hf_hub_download(
            repo_id="mauro-nievoff/MultiCaRe_Dataset",
            filename="cases.parquet",
            repo_type="dataset",
            local_dir=str(output_dir),
        )
        cases_df = pd.read_parquet(parquet_path)
        logger.info(
            "Loaded %d clinical cases via direct parquet download", len(cases_df)
        )
        cases_df.to_parquet(cache_path, index=False)
        return cases_df

    except Exception as e:
        raise RuntimeError(
            f"All MultiCaRe download methods failed. Last error: {e}\n"
            "Please check your internet connection and try again."
        ) from e


def _detect_case_text_column(df: pd.DataFrame) -> str:
    """Find the column containing clinical case text."""
    candidates = [
        "case", "case_text", "clinical_case", "text",
        "case_description", "content", "narrative",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        if "case" in col.lower() and df[col].dtype == object:
            sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""
            if len(sample) > 100:
                return col

    raise ValueError(
        f"Could not detect case text column. Columns found: {list(df.columns)}"
    )


def filter_multicare_for_mmcqsd(
    cases_df: pd.DataFrame,
    min_case_length: int = 50,
) -> pd.DataFrame:
    """Filter MultiCaRe cases to those matching MMCQSD condition groups.

    Parameters
    ----------
    cases_df : pd.DataFrame
        Raw MultiCaRe cases dataframe.
    min_case_length : int
        Minimum word count for a case to be considered usable.

    Returns
    -------
    pd.DataFrame
        Filtered cases with condition group assignments.
    """
    case_col = _detect_case_text_column(cases_df)
    logger.info("Using column '%s' as case text source", case_col)

    cases_df = cases_df.copy()
    cases_df["case_text_clean"] = cases_df[case_col].astype(str).fillna("")

    # Drop empty or very short cases
    cases_df["word_count"] = cases_df["case_text_clean"].apply(
        lambda x: len(x.split())
    )
    before = len(cases_df)
    cases_df = cases_df[cases_df["word_count"] >= min_case_length].copy()
    logger.info(
        "Kept %d / %d cases with >= %d words",
        len(cases_df), before, min_case_length,
    )

    # Assign condition groups
    logger.info("Assigning condition groups based on keyword matching...")
    tqdm.pandas(desc="Condition matching")
    cases_df["condition_groups"] = cases_df["case_text_clean"].progress_apply(
        _assign_condition_groups
    )

    # Keep only cases that match at least one condition
    cases_df["num_conditions"] = cases_df["condition_groups"].apply(len)
    matched = cases_df[cases_df["num_conditions"] > 0].copy()
    logger.info(
        "Found %d cases matching at least one MMCQSD condition group "
        "(out of %d total)",
        len(matched), len(cases_df),
    )

    # Explode into one row per condition group assignment
    exploded = matched.explode("condition_groups").copy()
    exploded = exploded.rename(columns={"condition_groups": "condition_group"})

    # Build clean output
    id_col = None
    for c in ["patient_id", "case_id", "pmcid", "article_id", "id"]:
        if c in exploded.columns:
            id_col = c
            break
    if id_col is None:
        exploded["case_id"] = [f"MC_{i:06d}" for i in range(len(exploded))]
        id_col = "case_id"

    output_cols = [id_col, "condition_group", "case_text_clean", "word_count"]
    for optional_col in ["pmcid", "article_id", "gender", "age", "image_type"]:
        if optional_col in exploded.columns and optional_col != id_col:
            output_cols.append(optional_col)

    result = exploded[output_cols].copy()
    result = result.rename(columns={"case_text_clean": "case_text", id_col: "case_id"})
    result = result.reset_index(drop=True)

    return result


def print_summary(filtered_df: pd.DataFrame) -> str:
    """Generate a human-readable summary of the filtering results."""
    lines = [
        "# MultiCaRe Filtering Summary",
        "",
        f"- Total filtered cases (exploded by condition): **{len(filtered_df)}**",
        f"- Unique cases: **{filtered_df['case_id'].nunique()}**",
        f"- Condition groups covered: **{filtered_df['condition_group'].nunique()}**",
        "",
        "## Per-Condition Breakdown",
        "",
        "| Condition Group | Cases | Avg Words |",
        "|---|---:|---:|",
    ]

    for cond, group in filtered_df.groupby("condition_group"):
        avg_words = group["word_count"].mean()
        lines.append(f"| {cond} | {len(group)} | {avg_words:.0f} |")

    lines.extend([
        "",
        "## Word Count Distribution",
        "",
        f"- Min: {filtered_df['word_count'].min()}",
        f"- Median: {filtered_df['word_count'].median():.0f}",
        f"- Mean: {filtered_df['word_count'].mean():.0f}",
        f"- Max: {filtered_df['word_count'].max()}",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and filter MultiCaRe for MMCQSD condition matching"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_OUTPUT_DIR,
        help="Directory for raw MultiCaRe download cache",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROCESSED_OUTPUT_DIR,
        help="Directory for processed output files",
    )
    parser.add_argument(
        "--min-case-length", type=int, default=50,
        help="Minimum word count for a case to be included",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Step 1: Download
    logger.info("Step 1/3: Downloading MultiCaRe cases...")
    cases_df = download_multicare_cases(args.raw_dir)
    logger.info("Downloaded %d total cases", len(cases_df))
    logger.info("Columns: %s", list(cases_df.columns))

    # Step 2: Filter
    logger.info("Step 2/3: Filtering to MMCQSD-relevant conditions...")
    filtered = filter_multicare_for_mmcqsd(
        cases_df, min_case_length=args.min_case_length
    )

    # Step 3: Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "multicare_filtered.csv"
    filtered.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved %d filtered rows to %s", len(filtered), output_path)

    summary_path = args.output_dir / "multicare_filter_summary.md"
    summary = print_summary(filtered)
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Summary saved to %s", summary_path)

    # Print summary to console
    print()
    print(summary)


if __name__ == "__main__":
    main()
