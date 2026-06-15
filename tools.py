"""
tools.py

The three required FitFindr tools. Each tool is a standalone function that
can be called and tested independently before being wired into the agent loop.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → str
    create_fit_card(outfit, new_item)               → str
"""

import os

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Scoring: keyword overlap against title, description, style_tags, and colors.
    Returns listings with score > 0, sorted by score descending.
    Returns [] if nothing matches — never raises.
    """
    listings = load_listings()

    # ── Step 1: filter by price and size ──────────────────────────────────────
    filtered = []
    for item in listings:
        if max_price is not None and item["price"] > max_price:
            continue
        if size is not None:
            item_size = item.get("size", "").lower()
            if size.lower() not in item_size:
                continue
        filtered.append(item)

    # ── Step 2: score by keyword overlap ─────────────────────────────────────
    keywords = set(description.lower().split())

    def score(item: dict) -> int:
        searchable = " ".join([
            item.get("title", ""),
            item.get("description", ""),
            " ".join(item.get("style_tags", [])),
            " ".join(item.get("colors", [])),
            item.get("category", ""),
            item.get("brand", "") or "",
        ]).lower()
        return sum(1 for kw in keywords if kw in searchable)

    scored = [(item, score(item)) for item in filtered]

    # ── Step 3: drop zero-score items and sort ────────────────────────────────
    results = [item for item, s in scored if s > 0]
    results.sort(key=lambda item: score(item), reverse=True)

    return results


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Handles empty wardrobe gracefully by offering general styling advice.
    Never raises an exception or returns an empty string.
    """
    try:
        client = _get_groq_client()
    except ValueError as e:
        return f"Could not connect to LLM: {e}. This {new_item.get('category', 'item')} would pair well with neutral basics."

    item_desc = (
        f"Item: {new_item.get('title', 'Unknown item')}\n"
        f"Category: {new_item.get('category', 'unknown')}\n"
        f"Style tags: {', '.join(new_item.get('style_tags', []))}\n"
        f"Colors: {', '.join(new_item.get('colors', []))}\n"
        f"Condition: {new_item.get('condition', 'unknown')}\n"
        f"Description: {new_item.get('description', '')}"
    )

    wardrobe_items = wardrobe.get("items", [])

    if not wardrobe_items:
        # Empty wardrobe — general styling advice
        prompt = (
            f"You're a thrift-savvy stylist. A user just found this secondhand piece:\n\n"
            f"{item_desc}\n\n"
            f"They have no wardrobe entered yet. Give 1–2 general outfit ideas: what kinds of "
            f"bottoms, layers, and shoes typically work well with this type of piece. "
            f"Be specific about silhouettes and vibes — not just 'jeans and sneakers'. "
            f"Keep it to 3–5 sentences total, casual tone."
        )
    else:
        wardrobe_text = "\n".join(
            f"- {w['name']} ({w['category']}, colors: {', '.join(w.get('colors', []))})"
            for w in wardrobe_items
        )
        prompt = (
            f"You're a thrift-savvy stylist. A user just found this secondhand piece:\n\n"
            f"{item_desc}\n\n"
            f"Their current wardrobe includes:\n{wardrobe_text}\n\n"
            f"Suggest 1–2 complete outfit combinations using the new item with specific pieces "
            f"from the wardrobe above. Name the exact wardrobe pieces. Include any styling tips "
            f"(tucking, layering, rolling sleeves, etc.). Casual tone, 4–6 sentences total."
        )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
        )
        result = response.choices[0].message.content.strip()
        return result if result else f"This {new_item.get('category', 'piece')} would pair well with high-waisted bottoms and minimal accessories."
    except Exception as e:
        return (
            f"Couldn't generate outfit suggestions right now, but this "
            f"{new_item.get('category', 'piece')} would pair well with neutral basics. "
            f"(Error: {e})"
        )


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Returns a 2–4 sentence Instagram/TikTok-style caption.
    Returns an error message string (not an exception) if outfit is empty.
    Uses temperature=0.9 for variety across calls.
    """
    # Guard against empty outfit
    if not outfit or not outfit.strip():
        return "Error: outfit description is required to generate a fit card."

    try:
        client = _get_groq_client()
    except ValueError as e:
        title = new_item.get("title", "this thrifted find")
        price = new_item.get("price", "?")
        platform = new_item.get("platform", "a thrift app")
        return f"just thrifted the {title} off {platform} for ${price} and i'm obsessed 🛍️"

    title = new_item.get("title", "this piece")
    price = new_item.get("price", "?")
    platform = new_item.get("platform", "a secondhand app")
    style_tags = ", ".join(new_item.get("style_tags", []))

    prompt = (
        f"Write a 2–4 sentence Instagram/TikTok OOTD caption for this thrifted outfit.\n\n"
        f"Thrifted item: {title}\n"
        f"Price: ${price}\n"
        f"Platform: {platform}\n"
        f"Vibe/style tags: {style_tags}\n"
        f"Outfit: {outfit}\n\n"
        f"Rules:\n"
        f"- Sound like a real person posting, not a brand or product description\n"
        f"- Mention the item name, price, and platform naturally once each\n"
        f"- Capture the specific vibe (not generic 'love this look!')\n"
        f"- Use lowercase, include 1–2 relevant emojis\n"
        f"- No hashtags\n"
        f"- 2–4 sentences only"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.9,
        )
        result = response.choices[0].message.content.strip()
        return result if result else f"thrifted this {title} off {platform} for ${price} and it's giving everything 🖤"
    except Exception as e:
        return f"thrifted this {title} off {platform} for ${price} and it's giving everything 🖤 (caption gen failed: {e})"