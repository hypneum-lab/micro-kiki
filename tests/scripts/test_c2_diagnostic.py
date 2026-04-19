"""Unit tests for scripts/c2_diagnostic.py — synthetic fixtures only."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _mock_run(queries):
    """Build a c2-downstream.json-shaped dict for one router from a list of queries.

    Each query dict must contain: expected_domain, routed_domain, correct_route,
    score, question, answer.
    """
    correct = [r["score"] for r in queries if r["correct_route"]]
    wrong = [r["score"] for r in queries if not r["correct_route"]]
    n = len(queries)
    return {
        "per_query": queries,
        "mean_score": sum(r["score"] for r in queries) / n,
        "routing_accuracy": sum(r["correct_route"] for r in queries) / n,
        "mean_score_when_routed_correct": sum(correct) / len(correct) if correct else 0.0,
        "mean_score_when_routed_wrong": sum(wrong) / len(wrong) if wrong else 0.0,
        "n_queries": n,
    }


def _mock_downstream(per_query_by_router: dict[str, list[dict]]):
    """Build a full c2-downstream.json-shaped dict across 3 routers."""
    return {
        "config": {"n_eval": len(per_query_by_router["oracle"]), "self_judging": True},
        "results": {router: _mock_run(qs) for router, qs in per_query_by_router.items()},
    }


def _q(domain, routed, score, question="q", answer="a"):
    return {
        "question": question,
        "expected_domain": domain,
        "routed_domain": routed,
        "correct_route": domain == routed,
        "score": score,
        "answer": answer,
    }


def test_per_domain_mean_gap_single_query_per_domain():
    from scripts.c2_diagnostic import analyze_per_domain

    # 3 domains × 1 query; oracle perfect (5), vqc routes wrong (1), random (3)
    data = _mock_downstream({
        "oracle": [_q("a", "a", 5), _q("b", "b", 5), _q("c", "c", 5)],
        "vqc":    [_q("a", "b", 1), _q("b", "c", 1), _q("c", "a", 1)],
        "random": [_q("a", "x", 3), _q("b", "x", 3), _q("c", "x", 3)],
    })
    out = analyze_per_domain(data, domains=["a", "b", "c"])
    assert set(out.keys()) == {"a", "b", "c"}
    for d in ["a", "b", "c"]:
        assert out[d]["mean_oracle"] == 5.0
        assert out[d]["mean_vqc"] == 1.0
        assert out[d]["mean_random"] == 3.0
        assert out[d]["gap_oracle_vs_vqc"] == 4.0
        assert out[d]["gap_vqc_vs_random"] == -2.0


def test_stratified_two_buckets_mutually_exclusive():
    from scripts.c2_diagnostic import analyze_stratified

    # 4 queries: vqc correct on 2, wrong on 2. oracle always correct.
    data = _mock_downstream({
        "oracle": [_q("a","a",5), _q("b","b",4), _q("c","c",4), _q("d","d",3)],
        "vqc":    [_q("a","a",3), _q("b","b",4), _q("c","x",1), _q("d","y",1)],
        "random": [_q("a","a",4), _q("b","x",2), _q("c","c",5), _q("d","x",3)],
    })
    out = analyze_stratified(data)
    assert out["vqc_correct"]["n"] == 2
    assert out["vqc_wrong"]["n"] == 2
    # In vqc_correct bucket (queries 0 and 1): vqc scores [3,4], mean=3.5
    assert out["vqc_correct"]["vqc_mean_score"] == 3.5
    # In vqc_wrong bucket (queries 2 and 3): vqc scores [1,1], mean=1.0
    assert out["vqc_wrong"]["vqc_mean_score"] == 1.0
    # Totals should cover all 4 queries
    assert out["vqc_correct"]["n"] + out["vqc_wrong"]["n"] == 4


def test_top_10_gap_sorted_desc_ties_stable():
    from scripts.c2_diagnostic import top_10_by_gap

    # 12 queries with distinct gaps + 2 ties at the boundary
    queries_oracle = [_q("a", "a", 5, question=f"q{i}") for i in range(12)]
    vqc_scores = [5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 2, 1]  # gaps = [0]*7 + [1,1,2,3,4]
    queries_vqc = [_q("a", "a", s, question=f"q{i}") for i, s in enumerate(vqc_scores)]
    data = _mock_downstream({
        "oracle": queries_oracle,
        "vqc":    queries_vqc,
        "random": [_q("a","a",3, question=f"q{i}") for i in range(12)],
    })
    out = top_10_by_gap(data, k=10)
    assert len(out) == 10
    # Top gap is q11 (oracle 5 - vqc 1 = 4)
    assert out[0]["question"] == "q11"
    assert out[0]["gap"] == 4
    # Q10 next (gap 3)
    assert out[1]["question"] == "q10"
    # Stable by input order for gap=0 ties
    assert all(o["gap"] >= 0 for o in out)


def test_full_pipeline_writes_json_and_figures(tmp_path):
    from scripts.c2_diagnostic import run

    # Minimal input: 2 domains × 3 queries each = 6 per router
    input_path = tmp_path / "c2-downstream.json"
    data = _mock_downstream({
        "oracle": [_q("a","a",5), _q("a","a",4), _q("a","a",5),
                   _q("b","b",3), _q("b","b",4), _q("b","b",5)],
        "vqc":    [_q("a","b",2), _q("a","a",3), _q("a","b",1),
                   _q("b","a",1), _q("b","b",4), _q("b","a",2)],
        "random": [_q("a","x",3), _q("a","y",3), _q("a","x",4),
                   _q("b","y",2), _q("b","x",3), _q("b","y",3)],
    })
    import json as _json
    input_path.write_text(_json.dumps(data))

    out_json = tmp_path / "c2-diagnostic.json"
    out_per_domain = tmp_path / "per-domain.pdf"
    out_stratified = tmp_path / "stratified.pdf"
    out_top10 = tmp_path / "top10.md"
    rc = run(
        input_path=input_path,
        out_json=out_json,
        out_per_domain_pdf=out_per_domain,
        out_stratified_pdf=out_stratified,
        out_top10_md=out_top10,
        domains=["a", "b"],
    )
    assert rc == 0
    assert out_json.exists()
    assert out_per_domain.exists()
    assert out_stratified.exists()
    assert out_top10.exists()
    report = _json.loads(out_json.read_text())
    assert set(report.keys()) == {"per_domain", "stratified", "top_gaps", "config"}
    assert set(report["per_domain"].keys()) == {"a", "b"}
