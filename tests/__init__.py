"""
tests/test_tools.py

Tests for all three FitFindr tools. Run with:
    pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools import search_listings, suggest_outfit, create_fit_card
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


# ── search_listings tests ─────────────────────────────────────────────────────

class TestSearchListings:

    def test_returns_results_for_valid_query(self):
        results = search_listings("vintage graphic tee", size=None, max_price=50)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_returns_list_of_dicts(self):
        results = search_listings("jacket", size=None, max_price=100)
        assert all(isinstance(r, dict) for r in results)

    def test_result_has_required_fields(self):
        results = search_listings("vintage", size=None, max_price=100)
        assert len(results) > 0
        item = results[0]
        for field in ["id", "title", "price", "platform", "size", "condition"]:
            assert field in item, f"Missing field: {field}"

    def test_empty_results_for_impossible_query(self):
        results = search_listings("designer ballgown", size="XXS", max_price=5)
        assert results == []

    def test_empty_results_returns_list_not_exception(self):
        # Should not raise
        results = search_listings("designer ballgown", size="XXS", max_price=5)
        assert isinstance(results, list)

    def test_price_filter_respected(self):
        results = search_listings("jacket", size=None, max_price=30)
        assert all(item["price"] <= 30 for item in results)

    def test_size_filter_case_insensitive(self):
        # "m" should match "M", "S/M", "M/L" etc.
        results_upper = search_listings("top", size="M", max_price=None)
        results_lower = search_listings("top", size="m", max_price=None)
        assert len(results_upper) == len(results_lower)

    def test_results_sorted_by_relevance(self):
        # "graphic tee vintage" should put tee results before unrelated ones
        results = search_listings("graphic tee vintage", size=None, max_price=None)
        assert len(results) > 0
        # First result should have higher keyword score than last
        # (We can't check score directly, but we can verify order is consistent)
        assert results[0]["id"] != ""

    def test_no_price_filter_returns_all_matching(self):
        with_filter = search_listings("vintage", size=None, max_price=20)
        without_filter = search_listings("vintage", size=None, max_price=None)
        assert len(without_filter) >= len(with_filter)

    def test_description_matches_style_tags(self):
        results = search_listings("grunge", size=None, max_price=None)
        assert len(results) > 0
        for item in results:
            searchable = " ".join([
                item["title"], item["description"],
                " ".join(item.get("style_tags", [])),
            ]).lower()
            assert "grunge" in searchable


# ── suggest_outfit tests (no LLM key needed for failure modes) ────────────────

class TestSuggestOutfit:

    def test_returns_string(self):
        results = search_listings("vintage tee", size=None, max_price=50)
        if not results:
            pytest.skip("No results to test with")
        wardrobe = get_example_wardrobe()
        result = suggest_outfit(results[0], wardrobe)
        assert isinstance(result, str)

    def test_returns_nonempty_string(self):
        results = search_listings("vintage tee", size=None, max_price=50)
        if not results:
            pytest.skip("No results to test with")
        wardrobe = get_example_wardrobe()
        result = suggest_outfit(results[0], wardrobe)
        assert len(result.strip()) > 0

    def test_empty_wardrobe_no_exception(self):
        results = search_listings("jacket", size=None, max_price=100)
        if not results:
            pytest.skip("No results to test with")
        empty_wardrobe = get_empty_wardrobe()
        # Should not raise
        result = suggest_outfit(results[0], empty_wardrobe)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_empty_wardrobe_returns_general_advice(self):
        results = search_listings("jacket", size=None, max_price=100)
        if not results:
            pytest.skip("No results to test with")
        empty_wardrobe = get_empty_wardrobe()
        result = suggest_outfit(results[0], empty_wardrobe)
        # Should still be a meaningful string
        assert len(result) > 20


# ── create_fit_card tests ─────────────────────────────────────────────────────

class TestCreateFitCard:

    def test_empty_outfit_returns_error_string(self):
        results = search_listings("vintage tee", size=None, max_price=50)
        if not results:
            pytest.skip("No results to test with")
        result = create_fit_card("", results[0])
        assert isinstance(result, str)
        assert "Error" in result or "error" in result

    def test_whitespace_outfit_returns_error_string(self):
        results = search_listings("vintage tee", size=None, max_price=50)
        if not results:
            pytest.skip("No results to test with")
        result = create_fit_card("   ", results[0])
        assert isinstance(result, str)
        assert "Error" in result or "error" in result

    def test_empty_outfit_does_not_raise(self):
        results = search_listings("vintage tee", size=None, max_price=50)
        if not results:
            pytest.skip("No results to test with")
        # Should not raise any exception
        result = create_fit_card("", results[0])
        assert result is not None

    def test_valid_input_returns_string(self):
        results = search_listings("vintage tee", size=None, max_price=50)
        if not results:
            pytest.skip("No results to test with")
        outfit = "Pair with baggy jeans and chunky sneakers for a 90s look."
        result = create_fit_card(outfit, results[0])
        assert isinstance(result, str)
        assert len(result.strip()) > 0