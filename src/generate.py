"""
Sentinel — Incident Report Generator
Loads the trained next-event LSTM, scores every test event, groups flagged
events into per-host incidents, and sends the top incidents to DeepSeek for
structured SOC analyst briefs. Writes incident_reports.jsonl.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import httpx
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.data import (
    ARGS_NUM_CLASSES,
    RETURN_VALUE_CLASSES,
    TrailingWindowDataset,
    _get_labels,
    _sort_within_hosts,
    build_categorical_vocab,
    build_process_name_vocab,
    build_vocab,
    load_host_splits,
    preprocess_features,
)
from src.model import NextEventLSTM
from src.train import collate_next_event


def _load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_scores(
    model: NextEventLSTM,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Per-event total surprisal for every scored test position."""
    model.eval()
    all_scores = []
    with torch.no_grad():
        for context, targets in loader:
            context = {k: v.to(device) for k, v in context.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            surprisal = model.surprisal(context, targets)
            all_scores.append(surprisal["total"].cpu())
    return torch.cat(all_scores).numpy()


def group_incidents(
    scores: np.ndarray,
    threshold: float,
    host_lengths: List[int],
    host_names: List[str],
    gap: int,
) -> List[dict]:
    """Group flagged events into per-host incidents, rank by peak score.

    Operates on scored-array positions (host_lengths = scored events per
    host). Caller is responsible for remapping to dataset positions.
    """
    flagged = scores >= threshold
    incidents: List[dict] = []
    offset = 0

    for host_idx, host_n in enumerate(host_lengths):
        host_flags = flagged[offset : offset + host_n]
        host_scores = scores[offset : offset + host_n]

        if not host_flags.any():
            offset += host_n
            continue

        # Find contiguous flagged runs
        runs: List[tuple[int, int]] = []
        i = 0
        while i < host_n:
            if host_flags[i]:
                start = i
                while i < host_n and host_flags[i]:
                    i += 1
                runs.append((start, i - 1))
            else:
                i += 1

        # Merge runs separated by <= gap+1 events
        merged: List[tuple[int, int]] = []
        for r_start, r_end in runs:
            if merged and (r_start - merged[-1][1] <= gap + 1):
                merged[-1] = (merged[-1][0], r_end)
            else:
                merged.append((r_start, r_end))

        for r_start, r_end in merged:
            loc = np.where(host_flags[r_start : r_end + 1])[0] + r_start
            incidents.append({
                "host": host_names[host_idx],
                "global_start": int(offset + r_start),
                "global_end": int(offset + r_end),
                "flagged_indices": [int(offset + p) for p in loc],
                "n_flagged": int(len(loc)),
                "peak_score": float(host_scores[r_start : r_end + 1].max()),
            })

        offset += host_n

    incidents.sort(key=lambda inc: inc["peak_score"], reverse=True)
    return incidents


def _extract_event_lines(
    test_df,
    indices: List[int],
    flagged_set: set,
) -> List[str]:
    """Convert dataset-position indices into one-line event strings."""
    lines = []
    for idx in indices:
        if idx < 0 or idx >= len(test_df):
            continue
        row = test_df.iloc[idx]
        event_str = (
            f"[{row.get('timestamp', '?')}] "
            f"processName={row.get('processName', '?')} "
            f"userId={row.get('userId', '?')} "
            f"argsNum={row.get('argsNum', '?')} "
            f"returnValue={row.get('returnValue', '?')} "
            f"eventId={row.get('eventId', '?')}"
        )
        if idx in flagged_set:
            event_str += "  *** FLAGGED ***"
        lines.append(event_str)
    return lines


def build_prompt(
    incident: dict,
    test_df,
    context_padding: int,
    max_events: int,
) -> str:
    """Construct the SOC analyst prompt for one incident."""
    gs, ge = incident["global_start"], incident["global_end"]
    flagged_set = set(incident["flagged_indices"])

    left = max(0, gs - context_padding)
    right = min(len(test_df) - 1, ge + context_padding)

    span = right - left + 1
    if span > max_events:
        excess = span - max_events
        left += excess // 2
        right -= excess - excess // 2

    window_indices = list(range(left, right + 1))
    event_lines = _extract_event_lines(test_df, window_indices, flagged_set)

    header = (
        f"Host: {incident['host']} | "
        f"{incident['n_flagged']} flagged, "
        f"{len(window_indices)} events shown "
        f"(positions {left}-{right})\n"
    )
    events_text = "\n".join(event_lines)

    return (
        "You are a SOC analyst. Review the following kernel process events "
        "flagged as malicious. Generate a structured incident brief with:\n"
        "1. Summary (what happened in 2-3 sentences)\n"
        "2. MITRE ATT&CK technique ID (most likely)\n"
        "3. Recommended action\n"
        "4. Confidence (0-1)\n\n"
        f"{header}\n{events_text}"
    )


def _call_deepseek(
    prompt: str,
    api_key: str,
    ds_config: dict,
) -> dict:
    """Send one prompt to DeepSeek with retry + exponential backoff."""
    max_retries = 3
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=120) as client:
                resp = client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": ds_config.get("model", "deepseek-chat"),
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Respond ONLY with valid JSON matching: "
                                    '{"summary":"...","mitre_technique":"T...",'
                                    '"recommended_action":"...","confidence":0.X}'
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": ds_config.get("max_tokens", 1024),
                        "temperature": ds_config.get("temperature", 0.3),
                        "response_format": {"type": "json_object"},
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise last_error  # type: ignore[misc]


def main(
    data_dir: str = "data/raw/per_host",
    output_dir: str = "outputs",
    config_path: str = "config.yaml",
    n_blocks: int = 50,
    tune_frac: float = 0.2,
    seed: int = 42,
    dry_run: bool = False,
):
    """Score, flag, group, rank, DeepSeek, write incident_reports.jsonl."""
    load_dotenv()

    cfg = _load_config(config_path)
    gen_cfg = cfg.get("generation", {})
    ds_cfg = cfg.get("deepseek", {})
    window_size = cfg.get("model", {}).get("window_size", 512)
    batch_size = cfg.get("model", {}).get("batch_size", 64)

    gap = int(gen_cfg.get("gap", 3))
    context_padding = int(gen_cfg.get("context_padding", 10))
    top_k = int(gen_cfg.get("top_k_incidents", 100))

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key and not dry_run:
        print("ERROR: DEEPSEEK_API_KEY not set. "
              "Set the env var or use --dry-run.")
        sys.exit(1)

    device = _get_device()
    print(f"Device: {device}")

    # ---- load data (same seed -> same split & carve-out) ----
    print(f"Loading data from {data_dir} (seed={seed}) ...")
    train_df, _val_df, _tune_df, test_df = load_host_splits(
        data_dir, n_blocks=n_blocks, tune_frac=tune_frac, seed=seed,
    )
    test_df, test_host_lengths = _sort_within_hosts(test_df)
    test_host_names = sorted(test_df["hostName"].unique())

    # Build vocabs from training data only
    train_df_sorted, _ = _sort_within_hosts(train_df)
    proc_vocab = build_process_name_vocab(train_df_sorted["processName"])
    args_vocab = build_vocab(train_df_sorted["args"], max_tokens=10_000)
    cat_vocabs = {
        "userId": build_categorical_vocab(train_df_sorted["userId"]),
        "eventId": build_categorical_vocab(train_df_sorted["eventId"]),
    }

    test_feat = preprocess_features(test_df, proc_vocab, args_vocab, cat_vocabs)
    test_ds = TrailingWindowDataset(
        test_feat, test_host_lengths, window_size, 1,
        labels=_get_labels(test_df),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_next_event,
    )

    n_evil = int(test_ds.labels[test_ds.centers].sum())
    print(f"  Test: {len(test_ds):,} scored events ({n_evil:,} evil)")

    # ---- load model ----
    model_path = Path(output_dir) / "model.pt"
    if not model_path.exists():
        print(f"ERROR: model.pt not found at {model_path}")
        sys.exit(1)

    print(f"Loading {model_path} ...")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = NextEventLSTM(**ckpt["init_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- load threshold ----
    eval_path = Path(output_dir) / "eval_results.json"
    if not eval_path.exists():
        print(f"ERROR: eval_results.json not found at {eval_path}")
        sys.exit(1)
    with open(eval_path) as f:
        eval_data = json.load(f)
    threshold = float(eval_data["threshold_tuning"]["threshold"])
    print(f"Flag threshold (attack-val): {threshold:.4f}")

    # ---- score test set ----
    print("Scoring test set ...")
    scores = _collect_scores(model, test_loader, device)

    # ---- group into incidents ----
    print(f"Grouping flagged events (gap={gap}) ...")
    scored_host_lengths = [
        max(0, hl - window_size) for hl in test_host_lengths
    ]
    incidents = group_incidents(
        scores, threshold, scored_host_lengths, test_host_names, gap,
    )
    n_flagged = int((scores >= threshold).sum())
    print(f"  {n_flagged:,} flagged -> {len(incidents)} incidents")
    if not incidents:
        print("No incidents found -- nothing to report.")
        return

    # Remap scored-array positions -> dataset positions for prompt building.
    centers_arr = np.array(test_ds.centers, dtype=np.int64)
    for inc in incidents:
        inc["global_start"] = int(centers_arr[inc["global_start"]])
        inc["global_end"] = int(centers_arr[inc["global_end"]])
        inc["flagged_indices"] = [
            int(centers_arr[i]) for i in inc["flagged_indices"]
        ]

    # ---- top-k ----
    before = len(incidents)
    incidents = incidents[:top_k]
    print(f"  Top-{top_k}: {len(incidents)} incidents (down from {before})")

    # ---- generate reports ----
    out_path = Path(output_dir) / "incident_reports.jsonl"
    mode = "DRY RUN" if dry_run else "DeepSeek API"
    print(f"\nGenerating reports ({mode}) ...")

    skipped: List[dict] = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, inc in enumerate(incidents):
            prompt = build_prompt(inc, test_df, context_padding, window_size)

            record: dict = {
                "incident_id": i,
                "host": inc["host"],
                "event_range": [inc["global_start"], inc["global_end"]],
                "n_flagged": inc["n_flagged"],
                "peak_score": inc["peak_score"],
                "flagged_indices": inc["flagged_indices"],
                "prompt": prompt,
            }

            if dry_run:
                record["llm"] = None
                record["error"] = "dry_run"
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                tag = f"peak={inc['peak_score']:.1f}"
                print(f"  [{i + 1}/{len(incidents)}] {inc['host']} ({tag})"
                      " - skipped")
                continue

            try:
                llm_resp = _call_deepseek(prompt, api_key, ds_cfg)
                record["llm"] = llm_resp
                record["error"] = None
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                tech = llm_resp.get("mitre_technique", "?")
                conf = llm_resp.get("confidence", "?")
                print(f"  [{i + 1}/{len(incidents)}] {inc['host']} "
                      f"({tech} conf={conf})")
            except Exception as exc:
                record["llm"] = None
                record["error"] = str(exc)
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                skipped.append({"incident_id": i, "host": inc["host"],
                                "error": str(exc)})
                print(f"  [{i + 1}/{len(incidents)}] {inc['host']} "
                      f"- SKIPPED: {exc}")

    # ---- summary ----
    ok = len(incidents) - len(skipped)
    print(f"\nDone. {ok} reports -> {out_path}")
    if skipped:
        print(f"  {len(skipped)} incidents skipped (API errors)")
        skip_path = Path(output_dir) / "skipped_incidents.json"
        with open(skip_path, "w") as f:
            json.dump(skipped, f, indent=2)
        print(f"  Skipped log -> {skip_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Sentinel - Incident Report Generator")
    p.add_argument("--data-dir", default="data/raw/per_host")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Skip DeepSeek API calls (test grouping)")
    p.add_argument("--top-k", type=int,
                   help="Override top_k_incidents from config")
    args = p.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        seed=args.seed,
        dry_run=args.dry_run,
    )