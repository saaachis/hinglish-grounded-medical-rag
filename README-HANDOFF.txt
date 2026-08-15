HANDOFF — Hinglish Grounded Medical RAG
=======================================
These are the data artifacts that are gitignored and therefore absent from
the GitHub repo. Extract at the ROOT of your clone, preserving paths.

    git clone <repo> && cd hinglish-grounded-medical-rag
    unzip handoff-tier1-essential.zip      # paths are already relative to root

Verify after extracting:

    python -c "import pandas as pd; d=pd.read_csv('data/processed/mmcqsd_multicare_paired.csv'); print(len(d), d.similarity_score.mean())"
    # expect: 3015 0.4996636...

    python -c "import pandas as pd; d=pd.read_csv('results/combined_h1h2/combined_scored.csv'); print(len(d))"
    # expect: 1165

WHAT EACH FILE UNBLOCKS
-----------------------
results/combined_h1h2/combined_scored.csv
    The 1,165 cached generations: pair_id, hinglish_query, both model outputs,
    all four scores, cmi_score, cmi_bucket.
    -> re-score with any new metric WITHOUT spending Groq quota
    -> re-run the H2 analysis on the two arms separately

data/processed/mmcqsd_multicare_paired.csv
    3,015 pairs. Join to the above on pair_id to recover `evidence_text`
    (needed for any evidence-based metric) and `english_summary`
    (the gold English query used for the H04 retrieval experiment).

data/faiss_index/evidence.index + evidence_metadata.csv
    The live 10,000-case index the demo uses. Enough to run retrieval
    evaluation and reproduce the Recall@k numbers. You do NOT need
    evidence_embeddings.npy unless you are rebuilding the index.

TIER 2 (separate zip)
---------------------
data/processed/multicare_filtered.csv    -> rebuild the index, build H3 corpora
data/processed/mmedbench_questions.csv   -> H3 exam-text corpus (use English subset only)

NOT INCLUDED — download these yourself, they are public
-------------------------------------------------------
data/processed/pubmedqa_records.csv  (465 MB, public; only ~2% is on-topic anyway)
data/raw/**                          (3.7 GB, all public sources)
results/h1_real_openi_mmcqsd/*.npy|*.index and results/dataset_comparison/**
    (600 MB of binaries from the abandoned Open-i approach — superseded, not needed)
