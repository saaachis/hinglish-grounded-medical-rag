"""Build a small HMG-mini dataset for H1 prototype testing."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT = "data/processed/hmg_mini.csv"

SEED_REPORTS = [
    {"report_id": "R001", "report_text": "Right-sided pleural effusion is present. Heart size is within normal limits."},
    {"report_id": "R002", "report_text": "Left lower lobe consolidation is noted, suspicious for pneumonia."},
    {"report_id": "R003", "report_text": "Mild cardiomegaly with pulmonary vascular congestion."},
    {"report_id": "R004", "report_text": "No acute cardiopulmonary abnormality detected."},
    {"report_id": "R005", "report_text": "Bibasilar atelectatic changes are seen."},
    {"report_id": "R006", "report_text": "Small left pleural effusion with adjacent basal atelectasis."},
    {"report_id": "R007", "report_text": "Right upper lobe infiltrate, likely infective etiology."},
    {"report_id": "R008", "report_text": "Patchy bilateral opacities consistent with multifocal pneumonia."},
    {"report_id": "R009", "report_text": "Cardiomediastinal silhouette is mildly enlarged."},
    {"report_id": "R010", "report_text": "No focal consolidation or pleural effusion."},
    {"report_id": "R011", "report_text": "Moderate right pleural effusion with compressive atelectasis."},
    {"report_id": "R012", "report_text": "Pulmonary edema pattern with vascular prominence."},
    {"report_id": "R013", "report_text": "Hyperinflated lungs; no acute infiltrate."},
    {"report_id": "R014", "report_text": "Left basilar opacity, could represent subsegmental atelectasis."},
    {"report_id": "R015", "report_text": "Mild bilateral pleural thickening without active disease."},
    {"report_id": "R016", "report_text": "Heart size is normal. Lungs are clear."},
    {"report_id": "R017", "report_text": "Dense right lower lobe consolidation with air bronchograms."},
    {"report_id": "R018", "report_text": "Perihilar interstitial markings suggest mild congestion."},
    {"report_id": "R019", "report_text": "Small bibasal infiltrates, correlate clinically for infection."},
    {"report_id": "R020", "report_text": "No pleural effusion, no pneumothorax, no focal opacity."},
]

QUERY_TEMPLATES = {
    "pleural_effusion": [
        "Doctor, right side chhati me pani bhar gaya hai kya?",
        "Report me pleural effusion dikh raha hai kya?",
        "Chest ke side me fluid jama hai kya please batao?",
    ],
    "pneumonia": [
        "Mujhe khansi aur bukhar hai, kya pneumonia ho sakta hai?",
        "Report ke hisaab se lungs me infection ya pneumonia lag raha hai kya?",
        "Doctor, consolidation ka matlab pneumonia hi hota hai kya?",
    ],
    "cardiomegaly": [
        "Heart size badha hua lag raha hai kya report me?",
        "Kya cardiomegaly mention hai aur ye serious hai?",
        "Doctor, dil thoda bada dikh raha hai kya X-ray me?",
    ],
    "no_acute": [
        "Kya report me koi serious dikkat nahi hai?",
        "No acute abnormality ka matlab sab normal hai kya?",
        "Doctor, report mostly normal hai kya confirm karo.",
    ],
    "atelectasis": [
        "Saas lene me problem hai, lungs me kuch collapse jaisa hai kya?",
        "Atelectasis ka mention hai kya report me?",
        "Lower lung thoda daba hua ya collapse type dikh raha hai kya?",
    ],
    "congestion": [
        "Pulmonary congestion ya edema jaisa kuch likha hai kya?",
        "Doctor, lungs me fluid overload ke signs dikh rahe hai kya?",
        "Vascular congestion ka meaning simple me batao.",
    ],
}

HINDI_TOKENS = {
    "kya",
    "hai",
    "me",
    "mujhe",
    "saas",
    "chhati",
    "pani",
    "bhar",
    "dikkat",
    "khansi",
    "bukhar",
    "ho",
    "sakta",
    "nahi",
}
COMMON_TOKENS = {
    "right",
    "left",
    "bilateral",
    "small",
    "mild",
    "moderate",
    "dense",
    "lower",
    "upper",
    "lobe",
    "pleural",
    "effusion",
    "consolidation",
    "pneumonia",
    "heart",
    "size",
    "normal",
    "lungs",
    "acute",
    "abnormality",
    "with",
    "without",
    "is",
    "are",
    "for",
    "likely",
    "pattern",
    "changes",
    "seen",
    "noted",
}


def _extract_anchor_terms(report_text: str) -> list[str]:
    lower = report_text.lower()
    anchors: list[str] = []
    for term in [
        "pleural effusion",
        "consolidation",
        "pneumonia",
        "cardiomegaly",
        "congestion",
        "atelectasis",
        "opacity",
        "infiltrate",
        "no acute",
    ]:
        if term in lower:
            anchors.append(term)
    return anchors


def _extract_unique_anchor_token(report_text: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", report_text.lower())
    candidates = [
        token for token in tokens
        if len(token) >= 7 and token not in COMMON_TOKENS
    ]
    return candidates[0] if candidates else ""


def _attach_anchor_hint(query: str, report_text: str) -> str:
    # Add a compact medical anchor to improve retrieval realism and specificity.
    anchors = _extract_anchor_terms(report_text)
    unique_token = _extract_unique_anchor_token(report_text)
    if not anchors:
        if unique_token and random.random() > 0.4:
            return f"{query} Report me '{unique_token}' term ka matlab kya hai?"
        return query

    anchor = random.choice(anchors)
    query = f"{query} Report me '{anchor}' mention hai kya?"
    if unique_token and random.random() > 0.5:
        query = f"{query} Aur '{unique_token}' ka meaning bhi batao."
    return query


def _detect_report_column(df: pd.DataFrame) -> str:
    candidates = ["report_text", "report", "findings", "impression", "text"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not detect report text column. Expected one of: "
        f"{', '.join(candidates)}"
    )


def _extract_query_template(report_text: str) -> str:
    lower = report_text.lower()
    side_hint = ""
    if "right" in lower:
        side_hint = "right side"
    elif "left" in lower:
        side_hint = "left side"
    elif "bilateral" in lower or "bibasal" in lower:
        side_hint = "dono side"

    severity_hint = ""
    if "mild" in lower:
        severity_hint = " halka"
    elif "moderate" in lower:
        severity_hint = " medium level"
    elif "dense" in lower:
        severity_hint = " kaafi zyada"

    if "effusion" in lower:
        return (
            random.choice(QUERY_TEMPLATES["pleural_effusion"])
            + (f" {side_hint} me" if side_hint else "")
            + (f"{severity_hint} lag raha hai kya?" if severity_hint else "")
        ).strip()
    if "pneumonia" in lower or "infect" in lower or "consolidation" in lower:
        return (
            random.choice(QUERY_TEMPLATES["pneumonia"])
            + (f" {side_hint} lung me" if side_hint else "")
            + (f"{severity_hint} hai kya?" if severity_hint else "")
        ).strip()
    if "cardiomegaly" in lower or "cardiomediastinal" in lower:
        return (
            random.choice(QUERY_TEMPLATES["cardiomegaly"])
            + (f" condition{severity_hint} hai kya?" if severity_hint else "")
        ).strip()
    if "atelect" in lower:
        return (
            random.choice(QUERY_TEMPLATES["atelectasis"])
            + (f" {side_hint} lower side pe" if side_hint else "")
            + (f"{severity_hint} issue hai kya?" if severity_hint else "")
        ).strip()
    if "congestion" in lower or "edema" in lower:
        return (
            random.choice(QUERY_TEMPLATES["congestion"])
            + (f" {side_hint} me" if side_hint else "")
            + (f"{severity_hint} symptoms hai kya?" if severity_hint else "")
        ).strip()
    if "no acute" in lower or "lungs are clear" in lower:
        return random.choice(QUERY_TEMPLATES["no_acute"])
    return "Doctor, meri report ka simple Hinglish meaning batao please."


def _estimate_cmi_bucket(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return "medium"
    hindi_like = sum(token in HINDI_TOKENS for token in tokens)
    ratio = hindi_like / len(tokens)
    if ratio < 0.25:
        return "low"
    if ratio < 0.5:
        return "medium"
    return "high"


def _build_from_seed(target_size: int) -> pd.DataFrame:
    rows: list[dict] = []
    for idx in range(target_size):
        report = SEED_REPORTS[idx % len(SEED_REPORTS)]
        query = _extract_query_template(report["report_text"])
        query = _attach_anchor_hint(query, report["report_text"])
        if random.random() > 0.5:
            query += " Thoda simple Hinglish me samjhao."
        if random.random() > 0.75:
            query = "Please explain: " + query
        rows.append(
            {
                "sample_id": f"S{idx + 1:04d}",
                "query_hinglish": query,
                "report_id": report["report_id"],
                "report_text": report["report_text"],
            }
        )
    return pd.DataFrame(rows)


def _build_from_input(input_path: Path, target_size: int) -> pd.DataFrame:
    source_df = pd.read_csv(input_path)
    report_col = _detect_report_column(source_df)

    if "report_id" not in source_df.columns:
        source_df["report_id"] = [
            f"R{idx + 1:05d}" for idx in range(len(source_df))
        ]

    source_df = source_df.dropna(subset=[report_col]).copy()
    source_df = source_df.head(target_size).copy()

    rows: list[dict] = []
    for idx, row in source_df.iterrows():
        report_text = str(row[report_col]).strip()
        query = _extract_query_template(report_text)
        query = _attach_anchor_hint(query, report_text)
        rows.append(
            {
                "sample_id": f"S{len(rows) + 1:04d}",
                "query_hinglish": query,
                "report_id": str(row["report_id"]),
                "report_text": report_text,
            }
        )
    return pd.DataFrame(rows)


def build_hmg_mini(
    output_path: Path,
    input_reports: Path | None = None,
    target_size: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)
    if input_reports is None:
        df = _build_from_seed(target_size)
    else:
        df = _build_from_input(input_reports, target_size)

    df["cmi_bucket"] = df["query_hinglish"].apply(_estimate_cmi_bucket)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HMG-mini dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Output CSV path",
    )
    parser.add_argument(
        "--input-reports",
        type=Path,
        default=None,
        help="Optional Open-i reports CSV path",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_hmg_mini(
        output_path=args.output,
        input_reports=args.input_reports,
        target_size=args.target_size,
        seed=args.seed,
    )
    print(f"Built HMG-mini at: {args.output}")
    print(f"Rows: {len(df)}")
    print(df["cmi_bucket"].value_counts().to_string())


if __name__ == "__main__":
    main()

