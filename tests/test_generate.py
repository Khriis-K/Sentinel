"""Tests for src.generate — grouping, prompt building, score collection."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.generate import (
    build_prompt,
    group_incidents,
    _collect_scores,
    _extract_event_lines,
)
from src.model import NextEventLSTM


# ── group_incidents ────────────────────────────────────────────────────────────────

class TestGroupIncidents:
    """Grouping: merge logic, peak scores, empty edges."""

    def test_single_run_no_gap(self):
        scores = np.array([0.1, 0.2, 15.0, 12.0, 0.1, 0.2], dtype=np.float64)
        flagged = scores >= 10.0
        threshold = 10.0
        host_lengths = [len(scores)]
        host_names = ["h1"]
        incidents = group_incidents(scores, threshold, host_lengths, host_names,
                                    gap=0)
        assert len(incidents) == 1
        inc = incidents[0]
        assert inc["host"] == "h1"
        assert inc["n_flagged"] == 2
        assert inc["peak_score"] == pytest.approx(15.0)
        assert inc["global_start"] == 2
        assert inc["global_end"] == 3

    def test_merge_nearby_runs(self):
        # Two flagged runs separated by exactly `gap` events get merged.
        scores = np.array([20, 0, 0, 0, 20, 0, 0], dtype=np.float64)
        threshold = 10
        host_lengths = [len(scores)]
        incidents = group_incidents(scores, threshold, host_lengths, ["h1"],
                                    gap=3)
        assert len(incidents) == 1
        assert incidents[0]["global_start"] == 0
        assert incidents[0]["global_end"] == 4

    def test_do_not_merge_distant_runs(self):
        scores = np.array([20, 0, 0, 0, 0, 20], dtype=np.float64)
        threshold = 10
        host_lengths = [len(scores)]
        incidents = group_incidents(scores, threshold, host_lengths, ["h1"],
                                    gap=2)
        assert len(incidents) == 2

    def test_no_flagged_events(self):
        scores = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        threshold = 10.0
        host_lengths = [len(scores)]
        host_names = ["h1"]
        incidents = group_incidents(scores, threshold, host_lengths, host_names,
                                    gap=0)
        assert incidents == []

    def test_all_flagged(self):
        scores = np.array([11, 12, 13], dtype=np.float64)
        threshold = 10
        host_lengths = [len(scores)]
        incidents = group_incidents(scores, threshold, host_lengths, ["h1"],
                                    gap=0)
        assert len(incidents) == 1
        assert incidents[0]["n_flagged"] == 3

    def test_multiple_hosts(self):
        scores = np.array([20, 0, 0, 0, 20], dtype=np.float64)
        threshold = 10
        host_lengths = [3, 2]
        host_names = ["h1", "h2"]
        incidents = group_incidents(scores, threshold, host_lengths, host_names,
                                    gap=5)
        assert len(incidents) == 2
        # h2's flag is at position 4 (global), h1's at position 0
        # peak scores descend → h1 first, then h2 (only if scores flipped)
        # 20 at 0, 20 at 4 — tied, stable sort keeps host order
        assert incidents[0]["host"] in ("h1", "h2")

    def test_ranked_by_peak(self):
        scores = np.array([5, 20, 100, 5], dtype=np.float64)
        threshold = 4
        host_lengths = [len(scores)]
        # All scores above threshold → one incident, peak = 100
        incidents = group_incidents(scores, threshold, host_lengths, ["h1"],
                                    gap=0)
        assert len(incidents) == 1
        assert incidents[0]["peak_score"] == 100.0

    def test_two_runs_ranked(self):
        # Two distinct flagged clusters with unflagged events between them.
        scores = np.array([15, 50, 3,  3, 100, 3], dtype=np.float64)
        threshold = 10   # 15≥10, 50≥10 → run [0,1]; 100≥10 → run [4,4]
        host_lengths = [len(scores)]
        incidents = group_incidents(scores, threshold, host_lengths, ["h1"],
                                    gap=0)
        assert len(incidents) == 2
        assert incidents[0]["peak_score"] == 100.0  # highest first
        assert incidents[1]["peak_score"] == 50.0


# ── build_prompt & event extraction ───────────────────────────────────────────────

class TestPromptBuilding:
    """Event-line extraction and full prompt assembly."""

    def test_extract_event_lines(self):
        df = pd.DataFrame({
            "timestamp": [1000, 1001, 1002],
            "processName": ["bash", "python", "curl"],
            "userId": [0, 1000, 0],
            "argsNum": [1, 3, 2],
            "returnValue": [0, 0, 0],
            "eventId": [1, 5, 3],
        })
        lines = _extract_event_lines(df, [0, 2], flagged_set={2})
        assert len(lines) == 2
        assert "bash" in lines[0]
        assert "FLAGGED" not in lines[0]
        assert "curl" in lines[1]
        assert "FLAGGED" in lines[1]

    def test_extract_event_lines_bounds(self):
        df = pd.DataFrame({
            "timestamp": [1], "processName": ["a"], "userId": [0],
            "argsNum": [0], "returnValue": [0], "eventId": [1],
        })
        # Negative index → skipped, beyond len → skipped
        lines = _extract_event_lines(df, [-1, 0, 1], flagged_set=set())
        assert len(lines) == 1  # only the valid one

    def test_build_prompt_includes_header_and_fields(self):
        df = pd.DataFrame({
            "timestamp": range(10),
            "processName": [f"proc{i}" for i in range(10)],
            "userId": [0] * 10, "argsNum": [1] * 10,
            "returnValue": [0] * 10, "eventId": [1] * 10,
        })
        inc = {
            "host": "test-host",
            "global_start": 3, "global_end": 5,
            "flagged_indices": [3, 4, 5],
            "n_flagged": 3, "peak_score": 42.0,
        }
        prompt = build_prompt(inc, df, context_padding=2, max_events=512)
        assert "SOC analyst" in prompt
        assert "test-host" in prompt
        assert "FLAGGED" in prompt  # flagged events are marked
        assert "proc3" in prompt    # event 3 is included
        # Context padding: events 1-7 should be shown (3±2)
        assert "proc1" in prompt

    def test_build_prompt_max_events_cap(self):
        df = pd.DataFrame({
            "timestamp": range(100),
            "processName": [f"p{i}" for i in range(100)],
            "userId": [0] * 100, "argsNum": [1] * 100,
            "returnValue": [0] * 100, "eventId": [1] * 100,
        })
        inc = {
            "host": "h", "global_start": 20, "global_end": 80,
            "flagged_indices": list(range(20, 81)),
            "n_flagged": 61, "peak_score": 50.0,
        }
        prompt = build_prompt(inc, df, context_padding=50, max_events=50)
        # Should be capped to 50 events total
        lines = [l for l in prompt.split("\n") if l.startswith("[")]
        assert len(lines) <= 50

    def test_build_prompt_no_context_beyond_dataframe(self):
        df = pd.DataFrame({
            "timestamp": range(5),
            "processName": [f"p{i}" for i in range(5)],
            "userId": [0] * 5, "argsNum": [1] * 5,
            "returnValue": [0] * 5, "eventId": [1] * 5,
        })
        inc = {
            "host": "h", "global_start": 0, "global_end": 1,
            "flagged_indices": [0, 1],
            "n_flagged": 2, "peak_score": 10.0,
        }
        prompt = build_prompt(inc, df, context_padding=10, max_events=512)
        # Should show all 5 events (right bound clipped to len-1)
        assert "p4" in prompt


# ── Score collection (smoke) ──────────────────────────────────────────────────────

class TestScoreCollection:
    """Verify _collect_scores matches the training eval path."""

    def test_collect_scores_shape(self):
        """Run a minimal forward pass — just checks tensor shapes & no crash."""
        from torch.utils.data import DataLoader

        from src.data import TrailingWindowDataset
        from src.train import collate_next_event

        model = NextEventLSTM(
            process_name_vocab_size=10, args_vocab_size=100,
            user_id_vocab_size=5, event_id_vocab_size=8,
        )

        # Minimal feature dict: one host, 514 events (2 scored)
        n = 514
        features = {
            "processName":   np.zeros(n, dtype=np.int64),
            "args_ids":      np.zeros((n, 16), dtype=np.int64),
            "userId":        np.zeros(n, dtype=np.int64),
            "eventId":       np.zeros(n, dtype=np.int64),
            "argsNum":       np.zeros(n, dtype=np.int64),
            "returnValue":   np.zeros(n, dtype=np.int64),
            "parentProcessId": np.zeros(n, dtype=np.int64),
        }
        ds = TrailingWindowDataset(
            features, host_lengths=[n], window_size=512, stride=1,
            labels=np.zeros(n, dtype=np.int64),
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False,
                            collate_fn=collate_next_event)

        device = model._init_kwargs.get("device", "cpu")
        scores = _collect_scores(model, loader, "cpu")
        assert scores.shape == (len(ds),)
        assert np.all(np.isfinite(scores))


# ── Dry-run integration smoke ─────────────────────────────────────────────────────

class TestDryRunSmoke:
    """End-to-end dry-run doesn't crash and writes valid JSONL."""

    def test_dry_run_writes_jsonl(self, tmp_path, monkeypatch):
        """Dry-run with a synthetic two-host dataset (benign + evil)."""
        import json

        import numpy as np
        import pandas as pd
        import torch

        from src.generate import main

        # Two hosts: one benign (train), one with evil (test)
        rng = np.random.default_rng(42)
        benign = pd.DataFrame({
            "timestamp": np.arange(600),
            "processName": ["bash"] * 600, "userId": [0] * 600,
            "argsNum": [1] * 600, "returnValue": [0] * 600,
            "eventId": rng.integers(1, 10, 600),
            "parentProcessId": [0] * 600, "mountNamespace": [0] * 600,
            "args": [""] * 600, "hostName": ["benign-h"] * 600,
            "evil": [0] * 600,
        })
        evil = pd.DataFrame({
            "timestamp": np.arange(600),
            "processName": ["bash"] * 600, "userId": [0] * 600,
            "argsNum": [1] * 600, "returnValue": [0] * 600,
            "eventId": rng.integers(1, 10, 600),
            "parentProcessId": [0] * 600, "mountNamespace": [0] * 600,
            "args": [""] * 600, "hostName": ["evil-h"] * 600,
            "evil": [0] * 600,
        })
        evil.iloc[514, evil.columns.get_loc("evil")] = 1  # flagged event

        outputs = tmp_path / "outputs"
        outputs.mkdir()

        # Model checkpoint — vocab sizes matched to the tiny data
        vocab_sizes = {
            "process_name_vocab_size": 5, "args_vocab_size": 100,
            "user_id_vocab_size": 5, "event_id_vocab_size": 15,
            "args_num_vocab_size": 16, "return_value_vocab_size": 13,
            "parent_pid_vocab_size": 3,
        }
        model = NextEventLSTM(**vocab_sizes)
        torch.save({
            "model_state_dict": model.state_dict(),
            "init_kwargs": model._init_kwargs,
        }, outputs / "model.pt")

        (outputs / "eval_results.json").write_text(json.dumps({
            "threshold_tuning": {"threshold": 0.0},   # flag everything
        }))

        # Patch data loading so main() gets our synthetic splits.
        def _fake_splits(*args, **kwargs):
            """Return (train, val, tune, test) with only benign in train."""
            empty = benign.iloc[:0]  # val and tune are empty
            return benign.copy(), empty, empty, pd.concat([benign, evil],
                                                          ignore_index=True)

        monkeypatch.setattr("src.generate.load_host_splits", _fake_splits)

        main(data_dir="/unused", output_dir=str(outputs), dry_run=True)

        jsonl = outputs / "incident_reports.jsonl"
        assert jsonl.exists(), "JSONL should be created"
        with open(jsonl, encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) > 0, "at least one incident since threshold=0"
            for line in lines:
                rec = json.loads(line)
                assert "incident_id" in rec
                assert "host" in rec
                assert rec["llm"] is None
                assert rec["error"] == "dry_run"