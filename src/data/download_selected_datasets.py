"""Download and convert selected datasets (Open-i + MMCQSD) for project use.

This script focuses on text/report-first downloads so the team can
start experimentation quickly without requiring large image assets.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

from datasets import Image, load_dataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


SOURCES = {
    "openi_primary": {
        "dataset_id": "ayyuce/Indiana_University_Chest_X-ray_Collection",
        "url": "https://huggingface.co/datasets/ayyuce/Indiana_University_Chest_X-ray_Collection",
    },
    "openi_fallback": {
        "dataset_id": "dz-osamu/IU-Xray",
        "url": "https://huggingface.co/datasets/dz-osamu/IU-Xray",
    },
    "mmcqsd": {
        "dataset_id": "ArkaAcharya/MMCQSD",
        "url": "https://huggingface.co/datasets/ArkaAcharya/MMCQSD",
    },
    "pubmedqa": {
        "dataset_id": "qitai123/PubMedQA",
        "url": "https://huggingface.co/datasets/qitai123/PubMedQA",
    },
    "mmedbench": {
        "dataset_id": "Henrychur/MMedBench",
        "url": "https://huggingface.co/datasets/Henrychur/MMedBench",
        "filename": "MMedBench.zip",
    },
}


def _safe_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        # Hugging Face Image(decode=False) often returns {"bytes": ..., "path": ...}
        path_value = value.get("path")
        if path_value:
            return str(path_value)
        if "bytes" in value:
            return "<image_binary_skipped>"
        compact = {k: v for k, v in value.items() if k != "bytes"}
        try:
            return json.dumps(compact, ensure_ascii=False)
        except Exception:
            return str(compact)
    if isinstance(value, list):
        # Preserve only lightweight image/path-like markers.
        compact_items: list[str] = []
        for item in value:
            if isinstance(item, dict):
                compact_items.append(_safe_cell(item))
            else:
                compact_items.append(str(item))
        return json.dumps(compact_items, ensure_ascii=False)
    # Hugging Face Image feature may decode to PIL objects; keep a stable marker.
    value_repr = repr(value)
    if "PIL." in value_repr or "PngImageFile" in value_repr:
        filename = getattr(value, "filename", None)
        return str(filename) if filename else "<image_available>"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _detect_column(columns: list[str], candidates: list[str]) -> str:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        for c_lower, original in lower_map.items():
            if cand in c_lower:
                return original
    return ""


def _stream_split_to_rows(
    dataset_id: str,
    split: str,
    max_rows: int,
    dataset_label: str,
    progress_every: int = 500,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    ds = load_dataset(dataset_id, split=split, streaming=True)
    features = getattr(ds, "features", {})
    # Avoid decoding large image binaries when we only need report/query text.
    for col_name, feat in features.items():
        if isinstance(feat, Image):
            ds = ds.cast_column(col_name, Image(decode=False))
    columns = list(getattr(ds, "features", {}).keys())
    for idx, row in enumerate(ds):
        rows.append(row)
        if (idx + 1) % progress_every == 0:
            logger.info("%s progress: %d rows fetched", dataset_label, idx + 1)
        if max_rows > 0 and idx + 1 >= max_rows:
            break
    if not columns and rows:
        columns = list(rows[0].keys())
    return rows, columns


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _safe_cell(row.get(c)) for c in columns})


def _prepare_openi_processed(raw_rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, str]]:
    report_col = _detect_column(columns, ["response", "report", "impression", "findings", "caption"])
    query_col = _detect_column(columns, ["query", "prompt", "question"])
    image_col = _detect_column(columns, ["image", "images"])
    id_col = _detect_column(columns, ["id", "uid", "study"])

    processed: list[dict[str, str]] = []
    for idx, row in enumerate(raw_rows):
        rid = _safe_cell(row.get(id_col)) if id_col else f"OPENI_{idx+1:06d}"
        report = _safe_cell(row.get(report_col)) if report_col else ""
        if not report.strip():
            continue
        processed.append(
            {
                "report_id": rid,
                "query_or_prompt": _safe_cell(row.get(query_col)) if query_col else "",
                "report_text": report,
                "image_reference": _safe_cell(row.get(image_col)) if image_col else "",
            }
        )
    return processed


def _prepare_openi_raw_text_only(
    raw_rows: list[dict[str, Any]],
    columns: list[str],
) -> list[dict[str, str]]:
    report_col = _detect_column(columns, ["response", "report", "impression", "findings", "caption"])
    query_col = _detect_column(columns, ["query", "prompt", "question"])
    image_col = _detect_column(columns, ["image", "images"])
    id_col = _detect_column(columns, ["id", "uid", "study"])

    compact_rows: list[dict[str, str]] = []
    for idx, row in enumerate(raw_rows):
        compact_rows.append(
            {
                "record_id": _safe_cell(row.get(id_col)) if id_col else f"OPENI_{idx+1:06d}",
                "question_or_prompt": _safe_cell(row.get(query_col)) if query_col else "",
                "report_text": _safe_cell(row.get(report_col)) if report_col else "",
                "image_reference": _safe_cell(row.get(image_col)) if image_col else "",
            }
        )
    return compact_rows


def _prepare_mmcqsd_processed(raw_rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, str]]:
    query_col = _detect_column(columns, ["query", "question", "code_mixed", "input"])
    summary_col = _detect_column(columns, ["summary", "target", "response", "output"])
    image_col = _detect_column(columns, ["image", "images", "img"])
    id_col = _detect_column(columns, ["id", "uid"])

    processed: list[dict[str, str]] = []
    for idx, row in enumerate(raw_rows):
        q = _safe_cell(row.get(query_col)) if query_col else ""
        s = _safe_cell(row.get(summary_col)) if summary_col else ""
        if not q.strip() and not s.strip():
            continue
        processed.append(
            {
                "sample_id": _safe_cell(row.get(id_col)) if id_col else f"MMCQSD_{idx+1:06d}",
                "hinglish_query": q,
                "english_summary_or_target": s,
                "image_reference": _safe_cell(row.get(image_col)) if image_col else "",
            }
        )
    return processed


def _stringify_pubmedqa_context(value: Any) -> str:
    if isinstance(value, dict):
        contexts = value.get("contexts") or []
        if isinstance(contexts, list):
            return " ".join(_safe_cell(item).strip() for item in contexts if _safe_cell(item).strip())
    if isinstance(value, list):
        return " ".join(_safe_cell(item).strip() for item in value if _safe_cell(item).strip())
    return _safe_cell(value)


def _stream_pubmedqa_rows(
    config_name: str,
    split: str,
    max_rows: int,
    progress_every: int = 500,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ds = load_dataset(SOURCES["pubmedqa"]["dataset_id"], config_name, split=split, streaming=True)
    for idx, row in enumerate(ds):
        row_copy = dict(row)
        row_copy["subset"] = config_name
        row_copy["split"] = split
        rows.append(row_copy)
        if (idx + 1) % progress_every == 0:
            logger.info("PubMedQA %s progress: %d rows fetched", config_name, idx + 1)
        if max_rows > 0 and idx + 1 >= max_rows:
            break
    return rows


def _prepare_pubmedqa_raw(raw_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact_rows: list[dict[str, str]] = []
    for idx, row in enumerate(raw_rows):
        compact_rows.append(
            {
                "record_id": _safe_cell(row.get("pubid")) or f"PUBMEDQA_{idx+1:06d}",
                "subset": _safe_cell(row.get("subset")),
                "split": _safe_cell(row.get("split")),
                "question": _safe_cell(row.get("question")),
                "context_text": _stringify_pubmedqa_context(row.get("context")),
                "long_answer": _safe_cell(row.get("long_answer")),
                "final_decision": _safe_cell(row.get("final_decision")),
            }
        )
    return compact_rows


def _prepare_pubmedqa_processed(raw_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    processed: list[dict[str, str]] = []
    for idx, row in enumerate(raw_rows):
        question = _safe_cell(row.get("question"))
        context_text = _stringify_pubmedqa_context(row.get("context"))
        long_answer = _safe_cell(row.get("long_answer"))
        if not question.strip() and not context_text.strip():
            continue
        processed.append(
            {
                "sample_id": _safe_cell(row.get("pubid")) or f"PUBMEDQA_{idx+1:06d}",
                "subset": _safe_cell(row.get("subset")),
                "split": _safe_cell(row.get("split")),
                "question": question,
                "context_text": context_text,
                "answer_rationale": long_answer,
                "final_decision": _safe_cell(row.get("final_decision")),
            }
        )
    return processed


def _load_mmedbench_rows(
    max_rows: int,
    progress_every: int = 500,
) -> list[dict[str, Any]]:
    zip_path = hf_hub_download(
        repo_id=SOURCES["mmedbench"]["dataset_id"],
        filename=SOURCES["mmedbench"]["filename"],
        repo_type="dataset",
    )
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        jsonl_names = [name for name in archive.namelist() if name.endswith(".jsonl")]
        for name in sorted(jsonl_names):
            parts = Path(name).parts
            split_name = parts[-2].lower() if len(parts) >= 2 else "unknown"
            language = Path(parts[-1]).stem
            with archive.open(name) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line.decode("utf-8"))
                    row["split"] = split_name
                    row["language"] = language
                    rows.append(row)
                    if len(rows) % progress_every == 0:
                        logger.info("MMedBench progress: %d rows fetched", len(rows))
                    if max_rows > 0 and len(rows) >= max_rows:
                        return rows
    return rows


def _prepare_mmedbench_raw(raw_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact_rows: list[dict[str, str]] = []
    for idx, row in enumerate(raw_rows):
        compact_rows.append(
            {
                "record_id": f"MMEDBENCH_{idx+1:06d}",
                "split": _safe_cell(row.get("split")),
                "language": _safe_cell(row.get("language")),
                "question": _safe_cell(row.get("question")),
                "options_json": _safe_cell(row.get("options")),
                "answer_text": _safe_cell(row.get("answer")),
                "answer_idx": _safe_cell(row.get("answer_idx")),
                "rationale": _safe_cell(row.get("rationale")),
                "meta_info": _safe_cell(row.get("meta_info")),
            }
        )
    return compact_rows


def _prepare_mmedbench_processed(raw_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    processed: list[dict[str, str]] = []
    for idx, row in enumerate(raw_rows):
        question = _safe_cell(row.get("question"))
        answer = _safe_cell(row.get("answer"))
        rationale = _safe_cell(row.get("rationale"))
        if not question.strip():
            continue
        processed.append(
            {
                "sample_id": f"MMEDBENCH_{idx+1:06d}",
                "split": _safe_cell(row.get("split")),
                "language": _safe_cell(row.get("language")),
                "question": question,
                "options_json": _safe_cell(row.get("options")),
                "answer_text": answer,
                "answer_idx": _safe_cell(row.get("answer_idx")),
                "rationale": rationale,
                "meta_info": _safe_cell(row.get("meta_info")),
            }
        )
    return processed


def _write_source_manifest(path: Path) -> None:
    manifest = {
        "selected_datasets": {
            "openi_primary": SOURCES["openi_primary"],
            "openi_fallback": SOURCES["openi_fallback"],
            "mmcqsd": SOURCES["mmcqsd"],
            "pubmedqa": SOURCES["pubmedqa"],
            "mmedbench": SOURCES["mmedbench"],
        },
        "notes": [
            "Selected datasets are fetched via stable Hugging Face access routes for scripted downloading.",
            "Image-heavy assets are intentionally skipped in this phase; report/query text is prioritized.",
            "MMedBench is downloaded as the official zip archive and parsed directly because the default dataset loader has schema inconsistencies across files.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run(
    data_root: Path,
    openi_max_rows: int,
    mmcqsd_max_rows: int,
    pubmedqa_max_rows: int,
    mmedbench_max_rows: int,
    use_openi_fallback: bool,
    skip_openi: bool,
    skip_mmcqsd: bool,
    skip_pubmedqa: bool,
    skip_mmedbench: bool,
) -> None:
    raw_dir = data_root / "raw"
    processed_dir = data_root / "processed"
    raw_openi_dir = raw_dir / "openi"
    raw_mmcqsd_dir = raw_dir / "mmcqsd"
    raw_pubmedqa_dir = raw_dir / "pubmedqa"
    raw_mmedbench_dir = raw_dir / "mmedbench"
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_openi_dir.mkdir(parents=True, exist_ok=True)
    raw_mmcqsd_dir.mkdir(parents=True, exist_ok=True)
    raw_pubmedqa_dir.mkdir(parents=True, exist_ok=True)
    raw_mmedbench_dir.mkdir(parents=True, exist_ok=True)

    openi_source = SOURCES["openi_fallback"] if use_openi_fallback else SOURCES["openi_primary"]
    if not skip_openi:
        logger.info("Using Open-i source: %s", openi_source["dataset_id"])
        openi_rows, openi_cols = _stream_split_to_rows(
            dataset_id=openi_source["dataset_id"],
            split="train",
            max_rows=openi_max_rows,
            dataset_label="Open-i",
        )
        openi_raw_csv = raw_openi_dir / "openi_train_raw.csv"
        openi_raw_compact = _prepare_openi_raw_text_only(openi_rows, openi_cols)
        _write_csv(
            openi_raw_csv,
            openi_raw_compact,
            ["record_id", "question_or_prompt", "report_text", "image_reference"],
        )
        openi_processed = _prepare_openi_processed(openi_rows, openi_cols)
        openi_processed_csv = processed_dir / "openi_reports.csv"
        _write_csv(
            openi_processed_csv,
            openi_processed,
            ["report_id", "query_or_prompt", "report_text", "image_reference"],
        )
        logger.info("Open-i rows saved (raw): %d", len(openi_rows))
        logger.info("Open-i rows saved (processed): %d", len(openi_processed))

    if not skip_mmcqsd:
        logger.info("Using MMCQSD source: %s", SOURCES["mmcqsd"]["dataset_id"])
        mmcqsd_rows, mmcqsd_cols = _stream_split_to_rows(
            dataset_id=SOURCES["mmcqsd"]["dataset_id"],
            split="train",
            max_rows=mmcqsd_max_rows,
            dataset_label="MMCQSD",
        )
        mmcqsd_raw_csv = raw_mmcqsd_dir / "mmcqsd_train_raw.csv"
        _write_csv(mmcqsd_raw_csv, mmcqsd_rows, mmcqsd_cols)
        mmcqsd_processed = _prepare_mmcqsd_processed(mmcqsd_rows, mmcqsd_cols)
        mmcqsd_processed_csv = processed_dir / "mmcqsd_queries.csv"
        _write_csv(
            mmcqsd_processed_csv,
            mmcqsd_processed,
            ["sample_id", "hinglish_query", "english_summary_or_target", "image_reference"],
        )
        logger.info("MMCQSD rows saved (raw): %d", len(mmcqsd_rows))
        logger.info("MMCQSD rows saved (processed): %d", len(mmcqsd_processed))

    if not skip_pubmedqa:
        logger.info("Using PubMedQA source: %s", SOURCES["pubmedqa"]["dataset_id"])
        pubmedqa_rows: list[dict[str, Any]] = []
        for config_name in ("pqa_labeled", "pqa_unlabeled", "pqa_artificial"):
            pubmedqa_rows.extend(
                _stream_pubmedqa_rows(
                    config_name=config_name,
                    split="train",
                    max_rows=pubmedqa_max_rows,
                )
            )
        pubmedqa_raw_csv = raw_pubmedqa_dir / "pubmedqa_all_raw.csv"
        pubmedqa_raw = _prepare_pubmedqa_raw(pubmedqa_rows)
        _write_csv(
            pubmedqa_raw_csv,
            pubmedqa_raw,
            ["record_id", "subset", "split", "question", "context_text", "long_answer", "final_decision"],
        )
        pubmedqa_processed = _prepare_pubmedqa_processed(pubmedqa_rows)
        pubmedqa_processed_csv = processed_dir / "pubmedqa_records.csv"
        _write_csv(
            pubmedqa_processed_csv,
            pubmedqa_processed,
            ["sample_id", "subset", "split", "question", "context_text", "answer_rationale", "final_decision"],
        )
        logger.info("PubMedQA rows saved (raw): %d", len(pubmedqa_raw))
        logger.info("PubMedQA rows saved (processed): %d", len(pubmedqa_processed))

    if not skip_mmedbench:
        logger.info("Using MMedBench source: %s", SOURCES["mmedbench"]["dataset_id"])
        mmedbench_rows = _load_mmedbench_rows(max_rows=mmedbench_max_rows)
        mmedbench_raw_csv = raw_mmedbench_dir / "mmedbench_all_raw.csv"
        mmedbench_raw = _prepare_mmedbench_raw(mmedbench_rows)
        _write_csv(
            mmedbench_raw_csv,
            mmedbench_raw,
            ["record_id", "split", "language", "question", "options_json", "answer_text", "answer_idx", "rationale", "meta_info"],
        )
        mmedbench_processed = _prepare_mmedbench_processed(mmedbench_rows)
        mmedbench_processed_csv = processed_dir / "mmedbench_questions.csv"
        _write_csv(
            mmedbench_processed_csv,
            mmedbench_processed,
            ["sample_id", "split", "language", "question", "options_json", "answer_text", "answer_idx", "rationale", "meta_info"],
        )
        logger.info("MMedBench rows saved (raw): %d", len(mmedbench_raw))
        logger.info("MMedBench rows saved (processed): %d", len(mmedbench_processed))

    _write_source_manifest(processed_dir / "dataset_sources_manifest.json")
    logger.info("Source manifest written to processed folder.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected datasets for prototype")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--openi-max-rows", type=int, default=2000)
    parser.add_argument("--mmcqsd-max-rows", type=int, default=3015)
    parser.add_argument("--pubmedqa-max-rows", type=int, default=0)
    parser.add_argument("--mmedbench-max-rows", type=int, default=0)
    parser.add_argument(
        "--use-openi-fallback",
        action="store_true",
        help="Use dz-osamu/IU-Xray instead of primary Open-i route",
    )
    parser.add_argument(
        "--skip-openi",
        action="store_true",
        help="Skip Open-i download",
    )
    parser.add_argument(
        "--skip-mmcqsd",
        action="store_true",
        help="Skip MMCQSD download",
    )
    parser.add_argument(
        "--skip-pubmedqa",
        action="store_true",
        help="Skip PubMedQA download",
    )
    parser.add_argument(
        "--skip-mmedbench",
        action="store_true",
        help="Skip MMedBench download",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    run(
        data_root=args.data_root,
        openi_max_rows=args.openi_max_rows,
        mmcqsd_max_rows=args.mmcqsd_max_rows,
        pubmedqa_max_rows=args.pubmedqa_max_rows,
        mmedbench_max_rows=args.mmedbench_max_rows,
        use_openi_fallback=args.use_openi_fallback,
        skip_openi=args.skip_openi,
        skip_mmcqsd=args.skip_mmcqsd,
        skip_pubmedqa=args.skip_pubmedqa,
        skip_mmedbench=args.skip_mmedbench,
    )


if __name__ == "__main__":
    main()

