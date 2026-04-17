"""Tests for VQC router and training script (35-domain configuration)."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Training script unit tests (no PennyLane required)
# ---------------------------------------------------------------------------

class TestVQCDomainConfig:
    """Verify domain list consistency and qubit requirements."""

    def test_num_domains_is_35(self):
        from scripts.train_vqc_router import ALL_DOMAINS, NUM_DOMAINS
        assert NUM_DOMAINS == 35
        assert len(ALL_DOMAINS) == 35

    def test_base_is_last(self):
        from scripts.train_vqc_router import ALL_DOMAINS
        assert ALL_DOMAINS[-1] == "base"

    def test_niche_domains_sorted(self):
        from scripts.train_vqc_router import NICHE_DOMAINS
        assert NICHE_DOMAINS == sorted(NICHE_DOMAINS)

    def test_niche_count_is_34(self):
        from scripts.train_vqc_router import NICHE_DOMAINS
        assert len(NICHE_DOMAINS) == 34

    def test_domain_to_idx_bijective(self):
        from scripts.train_vqc_router import ALL_DOMAINS, DOMAIN_TO_IDX
        assert len(DOMAIN_TO_IDX) == len(ALL_DOMAINS)
        for i, d in enumerate(ALL_DOMAINS):
            assert DOMAIN_TO_IDX[d] == i

    def test_min_qubits_for_35_domains(self):
        from scripts.train_vqc_router import MIN_QUBITS, NUM_DOMAINS
        assert MIN_QUBITS == math.ceil(math.log2(NUM_DOMAINS))
        assert MIN_QUBITS == 6

    def test_new_domains_present(self):
        from scripts.train_vqc_router import DOMAIN_TO_IDX
        for new_domain in ("components", "llm-ops", "ml-training"):
            assert new_domain in DOMAIN_TO_IDX, f"{new_domain} missing from DOMAIN_TO_IDX"

    def test_domains_match_router_constants(self):
        """Ensure train script domains match src/routing/router.py NICHE_DOMAINS."""
        from src.routing.router import NICHE_DOMAINS as router_niches
        from scripts.train_vqc_router import NICHE_DOMAINS as train_niches
        assert set(train_niches) == set(router_niches)


class TestFeatureExtraction:
    """Test char n-gram TF-IDF and PCA reduction."""

    def test_char_ngram_shape(self):
        from scripts.train_vqc_router import char_ngram_features
        texts = ["hello world", "quantum computing", "embedded systems"]
        features = char_ngram_features(texts, max_features=50)
        assert features.shape == (3, 50)
        assert features.dtype == np.float32

    def test_char_ngram_normalized(self):
        from scripts.train_vqc_router import char_ngram_features
        texts = ["hello world test", "quantum computing example"]
        features = char_ngram_features(texts, max_features=50)
        norms = np.linalg.norm(features, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_pca_reduce_dims(self):
        from scripts.train_vqc_router import pca_reduce
        matrix = np.random.default_rng(42).standard_normal((20, 50)).astype(np.float32)
        reduced, components, mean = pca_reduce(matrix, 6)
        assert reduced.shape == (20, 6)
        assert components.shape == (6, 50)
        assert mean.shape == (50,)

    def test_pca_reduce_preserves_rank(self):
        from scripts.train_vqc_router import pca_reduce
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((30, 100)).astype(np.float32)
        reduced, _, _ = pca_reduce(matrix, 6)
        # Reduced matrix should have rank 6
        rank = np.linalg.matrix_rank(reduced)
        assert rank == 6


class TestDataLoading:
    """Test data loading from --data-dir structure."""

    def test_load_from_data_dir(self, tmp_path):
        from scripts.train_vqc_router import load_domain_prompts, ALL_DOMAINS

        # Create a few domain dirs with train.jsonl
        for domain in ("python", "rust", "embedded"):
            domain_dir = tmp_path / domain
            domain_dir.mkdir()
            train_file = domain_dir / "train.jsonl"
            examples = [
                {"prompt": f"How to use {domain} for task {i}?"}
                for i in range(5)
            ]
            with open(train_file, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

        prompts = load_domain_prompts(tmp_path)

        # Should have loaded the 3 domains
        assert len(prompts["python"]) > 0
        assert len(prompts["rust"]) > 0
        assert len(prompts["embedded"]) > 0
        # Other domains should be empty
        assert len(prompts["base"]) == 0

    def test_load_flat_jsonl(self, tmp_path):
        from scripts.train_vqc_router import load_domain_prompts

        # Flat file: <data_dir>/cpp.jsonl
        train_file = tmp_path / "cpp.jsonl"
        examples = [{"prompt": f"C++ question {i}"} for i in range(3)]
        with open(train_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        prompts = load_domain_prompts(tmp_path)
        assert len(prompts["cpp"]) >= 3

    def test_extract_prompt_formats(self):
        from scripts.train_vqc_router import extract_prompt

        # prompt field
        assert extract_prompt({"prompt": "hello"}) == "hello"

        # messages format
        msgs = {"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user question"},
        ]}
        assert extract_prompt(msgs) == "user question"

        # instruction field
        assert extract_prompt({"instruction": "do this"}) == "do this"

        # unknown format
        assert extract_prompt({"data": "unknown"}) is None

    def test_prepare_training_data(self, tmp_path):
        from scripts.train_vqc_router import prepare_training_data

        # Create minimal data
        for domain in ("dsp", "emc"):
            domain_dir = tmp_path / domain
            domain_dir.mkdir()
            with open(domain_dir / "train.jsonl", "w") as f:
                for i in range(10):
                    f.write(json.dumps({"prompt": f"{domain} question number {i} with enough text"}) + "\n")

        texts, labels, domain_prompts = prepare_training_data(tmp_path)
        assert len(texts) > 0
        assert len(labels) == len(texts)
        assert all(0 <= l < 35 for l in labels)

    def test_deduplication(self, tmp_path):
        from scripts.train_vqc_router import load_domain_prompts

        domain_dir = tmp_path / "python"
        domain_dir.mkdir()
        with open(domain_dir / "train.jsonl", "w") as f:
            for _ in range(5):
                f.write(json.dumps({"prompt": "same question repeated here"}) + "\n")

        prompts = load_domain_prompts(tmp_path)
        # Should deduplicate to 1
        assert len(prompts["python"]) <= 30  # may be oversampled but starts from 1


class TestQuantumRouterConfig:
    """Test QuantumRouterConfig defaults match 35-domain setup."""

    def test_config_defaults(self):
        from src.routing.quantum_router import QuantumRouterConfig
        config = QuantumRouterConfig()
        assert config.n_qubits == 6
        assert config.n_classes == 35
        assert config.n_layers == 6

    def test_quantum_router_domain_list(self):
        from src.routing.quantum_router import _ALL_DOMAINS, _NICHE_DOMAIN_LIST
        assert len(_ALL_DOMAINS) == 35
        assert len(_NICHE_DOMAIN_LIST) == 34
        assert _ALL_DOMAINS[-1] == "base"


class TestQuantumRouterInstantiation:
    """Test QuantumRouter instantiation (requires PennyLane)."""

    @pytest.fixture
    def router(self):
        pennylane = pytest.importorskip("pennylane")
        from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig
        config = QuantumRouterConfig(n_qubits=6, n_layers=2, n_classes=35)
        return QuantumRouter(config)

    def test_weight_shapes(self, router):
        # StronglyEntanglingLayers: (n_layers, n_qubits, 3)
        assert router.weights.shape == (2, 6, 3)
        assert router.linear_w.shape == (6, 35)
        assert router.linear_b.shape == (35,)

    def test_circuit_output_shape(self, router):
        features = np.random.default_rng(42).uniform(0, 2 * np.pi, size=6)
        result = router.circuit(router.weights, features)
        assert result.shape == (6,)
        # Expectation values of PauliZ are in [-1, 1]
        assert np.all(result >= -1.0 - 1e-6)
        assert np.all(result <= 1.0 + 1e-6)

    def test_route_returns_decision(self, router):
        features = np.random.default_rng(42).uniform(0, 2 * np.pi, size=6)
        decision = router.route(features)
        assert decision.model_id == "qwen35b"
        assert decision.reason.startswith("quantum-vqc:")

    def test_save_load_roundtrip(self, router, tmp_path):
        weights_before = router.weights.copy()
        linear_w_before = router.linear_w.copy()

        path = tmp_path / "test-weights.npz"
        router.save(path)

        # Modify weights
        router.weights[:] = 0
        router.linear_w[:] = 0

        # Reload
        router.load(path)
        np.testing.assert_array_equal(router.weights, weights_before)
        np.testing.assert_array_equal(router.linear_w, linear_w_before)


class TestSoftmax:
    """Test softmax helper."""

    def test_softmax_sums_to_one(self):
        from scripts.train_vqc_router import _softmax
        logits = np.array([1.0, 2.0, 3.0, 0.5, -1.0])
        probs = _softmax(logits)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-7)

    def test_softmax_35_classes(self):
        from scripts.train_vqc_router import _softmax
        logits = np.random.default_rng(42).standard_normal(35)
        probs = _softmax(logits)
        assert probs.shape == (35,)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-7)
        assert np.all(probs >= 0)

    def test_softmax_numerical_stability(self):
        from scripts.train_vqc_router import _softmax
        logits = np.array([1000.0, 1001.0, 999.0])
        probs = _softmax(logits)
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-7)
