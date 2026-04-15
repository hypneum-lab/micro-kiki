from __future__ import annotations

import json
import pytest
from pathlib import Path
from src.eval.forgetting import check_forgetting, save_forgetting_report, ForgettingCheckResult

torch = pytest.importorskip("torch")
from src.eval.forgetting import compute_subspace_angle


class TestCheckForgetting:
    def test_passes_when_both_safe(self):
        r = check_forgetting(0.01, 45.0, "stack-01", "stack-02")
        assert r.passed is True

    def test_fails_when_both_bad(self):
        r = check_forgetting(0.05, 20.0, "stack-01", "stack-02")
        assert r.passed is False

    def test_passes_when_only_delta_bad(self):
        r = check_forgetting(0.05, 45.0, "stack-01", "stack-02")
        assert r.passed is True

    def test_passes_when_only_angle_bad(self):
        r = check_forgetting(0.01, 20.0, "stack-01", "stack-02")
        assert r.passed is True


class TestSubspaceAngle:
    def test_orthogonal_subspaces(self):
        a = torch.eye(768, 16)
        b = torch.zeros(768, 16)
        b[16:32, :] = torch.eye(16)
        angle = compute_subspace_angle(a, b)
        assert angle > 80.0  # near 90

    def test_identical_subspaces(self):
        a = torch.randn(768, 16)
        angle = compute_subspace_angle(a, a)
        assert angle < 1.0  # near 0


class TestSaveReport:
    def test_saves_json(self, tmp_path):
        results = [
            check_forgetting(0.01, 45.0, "stack-01", "stack-03"),
            check_forgetting(0.02, 50.0, "stack-02", "stack-03"),
        ]
        path = save_forgetting_report(results, "stack-03", str(tmp_path))
        data = json.loads(path.read_text())
        assert data["all_passed"] is True
        assert len(data["checks"]) == 2
