#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        timeout=90,
        max_retries=2,
    )


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    from langchain_core.messages import HumanMessage

    instructions = (
        "Read the attached supermarket receipt image. Return ONLY a JSON object "
        "with keys subtotal, rounding, paid, and discounts. Every monetary value "
        "must be a decimal string in HKD, without a dollar sign or commas. "
        "subtotal is the printed SUBTOTAL / 小計 after discounts but before "
        "ROUNDING. rounding is the signed amount on the ROUNDING line, or "
        "'0.00' if there is no such line. paid is the final amount charged "
        "after rounding, often on an OCTOPUS, VISA, or CASH line. Ignore "
        "change, card balance, points, dates, and duplicate payment details. "
        "discounts is an array of positive decimal strings: one for EACH "
        "negative discount, promotion, coupon, member, app, percentage-off, "
        "or packaging-damage amount printed in the item section BEFORE "
        "SUBTOTAL. Read the actual right-hand amount column. Do not count a "
        "discount twice when its amount also appears in its description. "
        "Never include ROUNDING in discounts. Include zero-valued discounts "
        "as zero or omit them. Example: a subtotal of 102.31, ROUNDING -0.01, "
        "OCTOPUS 102.30, and one 5%-off line -5.39 gives "
        '{"subtotal":"102.31","rounding":"-0.01","paid":"102.30",'
        '"discounts":["5.39"]}. No markdown or explanation.'
    )

    def parse_receipt(response: Any) -> tuple[Decimal, Decimal, Decimal]:
        content = response_text(response)
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"Model did not return JSON: {content[:200]!r}")
        data = json.loads(content[start : end + 1])

        def money(value: Any) -> Decimal:
            amount = Decimal(str(value).replace("HK$", "").replace("$", "").replace(",", ""))
            return amount.quantize(Decimal("0.01"))

        subtotal = money(data["subtotal"])
        rounding = money(data["rounding"])
        paid = money(data["paid"])
        discounts = data["discounts"]
        if not isinstance(discounts, list):
            raise ValueError("discounts must be a list")
        discount_total = sum((money(value) for value in discounts), Decimal("0.00"))
        if any(money(value) < 0 for value in discounts):
            raise ValueError("discount magnitudes must be positive")
        return paid, subtotal + discount_total, subtotal + rounding

    total_paid = Decimal("0.00")
    total_without_discounts = Decimal("0.00")
    for path in images:
        message = HumanMessage(content=[
            {"type": "text", "text": instructions},
            {"type": "image_url", "image_url": {"url": image_data_url(path)}},
        ])
        reply = chain.invoke([message])
        try:
            paid, original, calculated_paid = parse_receipt(reply)
        except (ValueError, KeyError, TypeError, InvalidOperation):
            retry = HumanMessage(content=[
                {"type": "text", "text": instructions + " Your previous answer could not be parsed; return valid JSON with all four keys."},
                {"type": "image_url", "image_url": {"url": image_data_url(path)}},
            ])
            paid, original, calculated_paid = parse_receipt(chain.invoke([retry]))
        if paid != calculated_paid:
            # Confirm inconsistent amounts against the image once more.
            check = HumanMessage(content=[
                {"type": "text", "text": instructions + " Verify the subtotal, ROUNDING, and final paid lines carefully; your previous values did not reconcile."},
                {"type": "image_url", "image_url": {"url": image_data_url(path)}},
            ])
            paid, original, _ = parse_receipt(chain.invoke([check]))
        total_paid += paid
        total_without_discounts += original

    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_without_discounts:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
