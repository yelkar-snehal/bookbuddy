import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.search import EMBEDDING_MODEL, search_books
from app.db.database import engine


ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = ROOT / "eval" / "retrieval_queries.json"
RESULTS_DIR = ROOT / "eval" / "results"
TOP_K = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def resolve_expected_book(session: Session, expected: dict) -> set[int]:
    # One expected book can have many rows, one for each edition.
    # Any of those rows counts as a hit.
    # Some names in the dataset contain repeated spaces ("Becky  Chambers"),
    # so whitespace is collapsed before matching.
    rows = session.execute(
        text(
            """
            SELECT books.id
            FROM books
            JOIN authors ON authors.id = books.author_id
            WHERE regexp_replace(books.title, '\\s+', ' ', 'g') ILIKE :title_pattern
                AND regexp_replace(authors.name, '\\s+', ' ', 'g') ILIKE :author_pattern
            """
        ),
        {
            "title_pattern": escape_like(expected["title"]) + "%",
            "author_pattern": "%" + escape_like(expected["author"]) + "%",
        },
    ).scalars().all()

    return set(rows)


def fetch_author_names(session: Session, author_ids: set[int]) -> dict[int, str]:
    if not author_ids:
        return {}

    rows = session.execute(
        text("SELECT id, name FROM authors WHERE id = ANY(:author_ids)"),
        {"author_ids": list(author_ids)},
    ).all()

    return {row.id: row.name for row in rows}


def evaluate_query(session: Session, query: dict, min_ratings: int | None) -> dict:
    expected_books = []
    unresolved = []

    for expected in query["expected"]:
        book_ids = resolve_expected_book(session, expected)

        if book_ids:
            expected_books.append({**expected, "book_ids": book_ids})
        else:
            unresolved.append(expected)

    # Call the API function directly, so the evaluation always measures
    # the same SQL and model that the endpoint uses.
    # Pass every argument: a missing one would get FastAPI's Query object
    # as its value instead of the default.
    results = search_books(
        q=query["query"],
        limit=TOP_K,
        min_ratings=min_ratings,
        db=session,
    )
    result_ids = [result.book.id for result in results]
    author_names = fetch_author_names(
        session,
        {result.book.author_id for result in results if result.book.author_id},
    )

    found = []
    for expected in expected_books:
        ranks = [
            rank
            for rank, book_id in enumerate(result_ids, start=1)
            if book_id in expected["book_ids"]
        ]
        if ranks:
            found.append({"title": expected["title"], "rank": min(ranks)})

    first_rank = min((item["rank"] for item in found), default=None)

    return {
        "id": query["id"],
        "type": query["type"],
        "query": query["query"],
        "expected_count": len(expected_books),
        "found": found,
        "unresolved": unresolved,
        "recall": len(found) / len(expected_books) if expected_books else None,
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
        "first_rank": first_rank,
        "top_results": [
            {
                "rank": rank,
                "book_id": result.book.id,
                "title": result.book.title,
                "author": author_names.get(result.book.author_id),
                "distance": round(result.distance, 4),
            }
            for rank, result in enumerate(results, start=1)
        ],
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(results: list[dict]) -> dict:
    # Queries with no expected book in the database cannot be scored.
    scored = [result for result in results if result["recall"] is not None]
    by_type = defaultdict(list)

    for result in scored:
        by_type[result["type"]].append(result)

    def metrics(items: list[dict]) -> dict:
        return {
            "queries": len(items),
            f"recall@{TOP_K}": round(mean([item["recall"] for item in items]), 3),
            "mrr": round(mean([item["reciprocal_rank"] for item in items]), 3),
            "hit_rate": round(mean([1.0 if item["found"] else 0.0 for item in items]), 3),
        }

    return {
        "overall": metrics(scored),
        "by_type": {query_type: metrics(items) for query_type, items in sorted(by_type.items())},
        "unscored_queries": [result["id"] for result in results if result["recall"] is None],
    }


def print_report(results: list[dict], summary: dict) -> None:
    print()
    print(f"{'id':<5} {'type':<11} {'recall':>6} {'rank':>5}  top result")
    print("-" * 90)

    for result in results:
        recall = "-" if result["recall"] is None else f"{result['recall']:.2f}"
        rank = result["first_rank"] or "miss"
        top = result["top_results"][0] if result["top_results"] else None
        top_text = f"{top['title'][:50]} ({top['author']})" if top else "(no results)"
        print(f"{result['id']:<5} {result['type']:<11} {recall:>6} {rank:>5}  {top_text}")

    print()
    print("Overall:", summary["overall"])
    for query_type, metrics in summary["by_type"].items():
        print(f"  {query_type:<11}", metrics)

    unresolved = [
        (result["id"], expected)
        for result in results
        for expected in result["unresolved"]
    ]
    if unresolved:
        print()
        print("Expected books not found in the database (fix the title or author in the query file):")
        for query_id, expected in unresolved:
            print(f"  {query_id}: {expected['title']} / {expected['author']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate semantic book search.")
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=None,
        help="Only return books with at least this many ratings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = json.loads(QUERIES_PATH.read_text())["queries"]
    logger.info(
        "Evaluating %d queries with %s, top %d, min_ratings=%s",
        len(queries),
        EMBEDDING_MODEL,
        TOP_K,
        args.min_ratings,
    )

    with Session(engine) as session:
        results = []
        for query in queries:
            logger.info("Running %s: %s", query["id"], query["query"])
            results.append(evaluate_query(session, query, args.min_ratings))

    summary = summarize(results)
    print_report(results, summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_label = f"min{args.min_ratings}" if args.min_ratings is not None else "nofilter"
    output_path = RESULTS_DIR / f"retrieval_{datetime.now():%Y%m%d_%H%M%S}_{run_label}.json"
    output_path.write_text(
        json.dumps(
            {
                "embedding_model": EMBEDDING_MODEL,
                "top_k": TOP_K,
                "min_ratings": args.min_ratings,
                "summary": summary,
                "results": results,
            },
            indent=2,
        )
    )
    logger.info("Saved results to %s", output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
