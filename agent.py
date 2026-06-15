"""
agent.py

The FitFindr planning loop. Orchestrates the three tools in response to a
natural language user query, passing state between them via a session dict.

Usage:
    from agent import run_agent
    from utils.data_loader import get_example_wardrobe

    result = run_agent(
        query="vintage graphic tee under $30, size M",
        wardrobe=get_example_wardrobe(),
    )
    print(result["fit_card"])
    print(result["error"])   # None on success
"""

import os
import re

from dotenv import load_dotenv

from tools import search_listings, suggest_outfit, create_fit_card

load_dotenv()


# ── session state ─────────────────────────────────────────────────────────────

def _new_session(query: str, wardrobe: dict) -> dict:
    """Initialize and return a fresh session dict for one user interaction."""
    return {
        "query": query,
        "parsed": {},
        "search_results": [],
        "selected_item": None,
        "wardrobe": wardrobe,
        "outfit_suggestion": None,
        "fit_card": None,
        "error": None,
    }


# ── query parser ──────────────────────────────────────────────────────────────

def _parse_query(query: str) -> dict:
    """
    Extract description, size, and max_price from a natural language query.

    Uses regex heuristics first (fast, no LLM call needed for common patterns),
    then falls back to simple keyword extraction. This keeps parsing snappy
    and avoids a round-trip just for parameter extraction.

    Returns:
        dict with keys: description (str), size (str|None), max_price (float|None)
    """
    # Extract price: "under $30", "$30", "30 dollars", "max $40"
    price_match = re.search(
        r'(?:under|max|below|less than)?\s*\$(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*dollars?',
        query, re.IGNORECASE
    )
    max_price = float(price_match.group(1) or price_match.group(2)) if price_match else None

    # Extract size: explicit "size X" or known size tokens (not year-like "90s")
    size_pattern = re.search(
        r'\bsize\s+([A-Z0-9/]+)\b|'
        r'\b(XXS|XXL|XS|XL|S/M|M/L|L/XL|[SM])\b(?!\s*[-/]\s*\d)',
        query, re.IGNORECASE
    )
    size = None
    if size_pattern:
        raw = (size_pattern.group(1) or size_pattern.group(2) or "").strip()
        if raw and not re.fullmatch(r'\d+(?:\.\d+)?', raw):
            size = raw.upper()

    # Description: strip price/size tokens and clean up
    desc = query
    if price_match:
        desc = desc[:price_match.start()] + desc[price_match.end():]
    if size_pattern:
        desc = desc[:size_pattern.start()] + desc[size_pattern.end():]
    # Strip filler phrases
    for filler in [
        r"i'm looking for", r"looking for", r"i want", r"find me", r"can you find",
        r"i need", r"searching for", r"help me find", r"show me",
        r"under", r"around", r"approximately", r"about",
    ]:
        desc = re.sub(filler, "", desc, flags=re.IGNORECASE)
    desc = re.sub(r'\s+', ' ', desc).strip(" ,.")

    return {
        "description": desc if desc else query,
        "size": size,
        "max_price": max_price,
    }


# ── planning loop ─────────────────────────────────────────────────────────────

def run_agent(query: str, wardrobe: dict) -> dict:
    """
    Main agent entry point. Runs the FitFindr planning loop for a single
    user interaction and returns the completed session dict.

    Args:
        query:    Natural language user request
        wardrobe: User's wardrobe dict

    Returns:
        Session dict. Check session["error"] first — if not None, the
        interaction ended early and outfit_suggestion / fit_card will be None.
    """
    # ── Step 1: Initialize session ────────────────────────────────────────────
    session = _new_session(query, wardrobe)

    # ── Step 2: Parse query → description, size, max_price ───────────────────
    parsed = _parse_query(query)
    session["parsed"] = parsed

    description = parsed["description"]
    size = parsed["size"]
    max_price = parsed["max_price"]

    # ── Step 3: Search listings ───────────────────────────────────────────────
    results = search_listings(description, size, max_price)
    session["search_results"] = results

    if not results:
        # Build a helpful error that tells the user what to try
        parts = [f'No listings found for "{description}"']
        if size:
            parts.append(f"size {size}")
        if max_price:
            parts.append(f"under ${max_price:.0f}")
        hint = "Try broader keywords"
        if size:
            hint += ", removing the size filter"
        if max_price:
            hint += ", or raising your price ceiling"
        session["error"] = f"{'. '.join(parts)}. {hint}."
        return session

    # ── Step 4: Select top result ─────────────────────────────────────────────
    session["selected_item"] = results[0]

    # ── Step 5: Suggest outfit ────────────────────────────────────────────────
    outfit_suggestion = suggest_outfit(results[0], wardrobe)
    session["outfit_suggestion"] = outfit_suggestion

    # ── Step 6: Create fit card ───────────────────────────────────────────────
    fit_card = create_fit_card(outfit_suggestion, results[0])
    session["fit_card"] = fit_card

    # ── Step 7: Return completed session ─────────────────────────────────────
    return session


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.data_loader import get_example_wardrobe, get_empty_wardrobe

    print("=== Happy path: graphic tee ===\n")
    session = run_agent(
        query="looking for a vintage graphic tee under $30",
        wardrobe=get_example_wardrobe(),
    )
    if session["error"]:
        print(f"Error: {session['error']}")
    else:
        print(f"Found: {session['selected_item']['title']}")
        print(f"\nOutfit: {session['outfit_suggestion']}")
        print(f"\nFit card: {session['fit_card']}")

    print("\n\n=== No-results path ===\n")
    session2 = run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
    )
    print(f"Error message: {session2['error']}")