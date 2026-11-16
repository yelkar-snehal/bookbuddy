
import argparse
import gc
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from lingua import Language, LanguageDetectorBuilder
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.orm import Session

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.db.database import engine


ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = ROOT / "eval" / "retrieval_queries.json"
RESULTS_DIR = ROOT / "eval" / "results"

TOP_K = 10

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BGE_MODEL = "BAAI/bge-small-en-v1.5"

BGE_TABLE = "benchmark_book_embeddings"
MINILM_TABLE = "book_embeddings"

HYBRID_CANDIDATES = 50
RRF_K = 60
SEMANTIC_WEIGHT = 2.0
KEYWORD_WEIGHT = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


BOOK_COLUMNS = """
    books.id,
    books.title,
    books.goodreads_id,
    books.url,
    books.author_id,
    books.star_rating,
    books.num_ratings,
    books.num_reviews,
    authors.name AS author
"""

MIN_RATINGS_FILTER = """
    (
        CAST(:min_ratings AS integer) IS NULL
        OR books.num_ratings >= :min_ratings
    )
"""


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def resolve_expected_book(
    session: Session,
    expected: dict,
) -> set[int]:
    rows = session.execute(
        text("""
            SELECT books.id
            FROM books
            JOIN authors ON authors.id = books.author_id
            WHERE regexp_replace(books.title, '\\s+', ' ', 'g')
                ILIKE :title_pattern
              AND regexp_replace(authors.name, '\\s+', ' ', 'g')
                ILIKE :author_pattern
        """),
        {
            "title_pattern": escape_like(expected["title"]) + "%",
            "author_pattern": "%" + escape_like(expected["author"]) + "%",
        },
    ).scalars().all()

    return set(rows)


def fetch_english_candidate_ids(
    session: Session,
) -> list[int]:
    logger.info("Fetching shared MiniLM/BGE candidate pool")

    rows = session.execute(
        text("""
            SELECT
                bge.book_id,
                bge.source_text
            FROM benchmark_book_embeddings bge
            JOIN book_embeddings mini
                ON mini.book_id = bge.book_id
            WHERE bge.embedding_model = :bge_model
              AND mini.embedding_model = :minilm_model
            ORDER BY bge.book_id
        """),
        {
            "bge_model": BGE_MODEL,
            "minilm_model": MINILM_MODEL,
        },
    ).mappings().all()

    logger.info(
        "Found %d books with embeddings from both models",
        len(rows),
    )

    logger.info("Detecting English-language books")

    detector = LanguageDetectorBuilder.from_all_languages().build()

    english_ids = []

    for row in rows:
        detected_language = detector.detect_language_of(
            row["source_text"]
        )

        if detected_language == Language.ENGLISH:
            english_ids.append(row["book_id"])

    logger.info(
        "English candidate pool: %d / %d books",
        len(english_ids),
        len(rows),
    )

    if not english_ids:
        raise ValueError("No English candidates found")

    return english_ids


def build_semantic_sql(table_name: str) -> str:
    return f"""
        SELECT
            {BOOK_COLUMNS},
            embeddings.embedding
                <=> CAST(:query_embedding AS vector) AS distance,
            NULL::double precision AS score
        FROM books
        JOIN authors ON authors.id = books.author_id
        JOIN {table_name} embeddings
            ON embeddings.book_id = books.id
        WHERE embeddings.embedding_model = :embedding_model
          AND embeddings.embedding_dimension = :embedding_dimension
          AND books.id = ANY(:candidate_ids)
          AND {MIN_RATINGS_FILTER}
        ORDER BY distance
        LIMIT :limit
    """


def build_hybrid_sql(table_name: str) -> str:
    return f"""
        WITH keyword_query AS (
            SELECT
                to_tsquery(
                    'simple',
                    replace(
                        CAST(plainto_tsquery('english', :q) AS text),
                        ' & ',
                        ' | '
                    )
                ) AS title_query,
                to_tsquery(
                    'simple',
                    replace(
                        CAST(plainto_tsquery('simple', :q) AS text),
                        ' & ',
                        ' | '
                    )
                ) AS author_query
        ),
        semantic_candidates AS (
            SELECT
                books.id,
                embeddings.embedding
                    <=> CAST(:query_embedding AS vector) AS distance
            FROM books
            JOIN {table_name} embeddings
                ON embeddings.book_id = books.id
            WHERE embeddings.embedding_model = :embedding_model
              AND embeddings.embedding_dimension = :embedding_dimension
              AND books.id = ANY(:candidate_ids)
              AND {MIN_RATINGS_FILTER}
            ORDER BY distance
            LIMIT :candidates
        ),
        semantic AS (
            SELECT
                id,
                distance,
                ROW_NUMBER() OVER (ORDER BY distance) AS rank
            FROM semantic_candidates
        ),
        keyword_matches AS (
            SELECT books.id
            FROM books
            CROSS JOIN keyword_query
            WHERE books.id = ANY(:candidate_ids)
              AND to_tsvector('english', books.title)
                  @@ keyword_query.title_query

            UNION

            SELECT books.id
            FROM authors
            JOIN books ON books.author_id = authors.id
            CROSS JOIN keyword_query
            WHERE books.id = ANY(:candidate_ids)
              AND to_tsvector('simple', authors.name)
                  @@ keyword_query.author_query
        ),
        keyword_candidates AS (
            SELECT
                books.id,
                ts_rank(
                    to_tsvector('english', books.title),
                    keyword_query.title_query
                )
                + ts_rank(
                    to_tsvector('simple', coalesce(authors.name, '')),
                    keyword_query.author_query
                ) AS keyword_score
            FROM keyword_matches
            JOIN books ON books.id = keyword_matches.id
            LEFT JOIN authors ON authors.id = books.author_id
            CROSS JOIN keyword_query
            WHERE {MIN_RATINGS_FILTER}
            ORDER BY
                keyword_score DESC,
                books.num_ratings DESC NULLS LAST
            LIMIT :candidates
        ),
        keyword AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    ORDER BY keyword_score DESC
                ) AS rank
            FROM keyword_candidates
        ),
        fused AS (
            SELECT
                id,
                semantic.distance,
                coalesce(
                    :semantic_weight / (:rrf_k + semantic.rank),
                    0
                )
                + coalesce(
                    :keyword_weight / (:rrf_k + keyword.rank),
                    0
                ) AS score
            FROM semantic
            FULL OUTER JOIN keyword USING (id)
        )
        SELECT
            {BOOK_COLUMNS},
            fused.distance,
            fused.score
        FROM fused
        JOIN books ON books.id = fused.id
        LEFT JOIN authors ON authors.id = books.author_id
        ORDER BY fused.score DESC
        LIMIT :limit
    """


def load_model(model_name: str) -> SentenceTransformer:
    logger.info("Loading model: %s", model_name)

    model = SentenceTransformer(model_name)

    logger.info("Model dimensions: %d", model.get_embedding_dimension())
    logger.info("Device: %s", model.device)

    return model


def encode_query(
    model: SentenceTransformer,
    model_name: str,
    query: str,
) -> list[float]:
    if model_name == BGE_MODEL:
        embedding = model.encode_query(
            query,
            normalize_embeddings=True,
        )
    else:
        embedding = model.encode(
            query,
            normalize_embeddings=True,
        )

    return embedding.tolist()


def search_candidates(
    session: Session,
    model: SentenceTransformer,
    model_name: str,
    table_name: str,
    query: dict,
    candidate_ids: list[int],
    min_ratings: int | None,
    mode: str,
    semantic_weight: float,
    keyword_weight: float,
) -> list[dict]:
    embedding_dimension = model.get_embedding_dimension()

    query_embedding = encode_query(
        model,
        model_name,
        query["query"],
    )

    if mode == "semantic":
        sql = build_semantic_sql(table_name)
    else:
        sql = build_hybrid_sql(table_name)

    rows = session.execute(
        text(sql),
        {
            "q": query["query"],
            "embedding_model": model_name,
            "embedding_dimension": embedding_dimension,
            "query_embedding": (
                "[" + ",".join(str(value) for value in query_embedding) + "]"
            ),
            "candidate_ids": candidate_ids,
            "limit": TOP_K,
            "min_ratings": min_ratings,
            "candidates": HYBRID_CANDIDATES,
            "rrf_k": RRF_K,
            "semantic_weight": semantic_weight,
            "keyword_weight": keyword_weight,
        },
    ).mappings().all()

    return [dict(row) for row in rows]


def evaluate_query(
    session: Session,
    model: SentenceTransformer,
    model_name: str,
    table_name: str,
    query: dict,
    candidate_ids: list[int],
    candidate_id_set: set[int],
    min_ratings: int | None,
    mode: str,
    semantic_weight: float,
    keyword_weight: float,
) -> dict:
    expected_books = []
    unresolved = []
    outside_candidate_pool = []

    for expected in query["expected"]:
        book_ids = resolve_expected_book(session, expected)

        if not book_ids:
            unresolved.append(expected)
            continue

        candidate_matches = book_ids.intersection(candidate_id_set)

        if not candidate_matches:
            outside_candidate_pool.append(expected)
        else:
            expected_books.append({
                **expected,
                "book_ids": candidate_matches,
            })

    results = search_candidates(
        session=session,
        model=model,
        model_name=model_name,
        table_name=table_name,
        query=query,
        candidate_ids=candidate_ids,
        min_ratings=min_ratings,
        mode=mode,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
    )

    result_ids = [result["id"] for result in results]

    found = []

    for expected in expected_books:
        ranks = [
            rank
            for rank, book_id in enumerate(result_ids, start=1)
            if book_id in expected["book_ids"]
        ]

        if ranks:
            found.append({
                "title": expected["title"],
                "rank": min(ranks),
            })

    first_rank = min(
        (item["rank"] for item in found),
        default=None,
    )

    return {
        "id": query["id"],
        "type": query["type"],
        "query": query["query"],
        # Only expected books that exist in the English candidate pool are
        # included in retrieval metrics. Out-of-pool references are tracked
        # separately as candidate-pool coverage, not retrieval failures.
        "expected_count": len(expected_books),
        "eligible_expected_count": len(expected_books),
        "outside_candidate_pool_count": len(outside_candidate_pool),
        "unresolved_count": len(unresolved),
        "found": found,
        "unresolved": unresolved,
        "outside_candidate_pool": outside_candidate_pool,
        "recall": (
            len(found) / len(expected_books)
            if expected_books
            else None
        ),
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
        "first_rank": first_rank,
        "top_results": [
            {
                "rank": rank,
                "book_id": result["id"],
                "title": result["title"],
                "author": result["author"],
                "distance": (
                    None
                    if result["distance"] is None
                    else round(float(result["distance"]), 4)
                ),
                "score": (
                    None
                    if result["score"] is None
                    else round(float(result["score"]), 5)
                ),
            }
            for rank, result in enumerate(results, start=1)
        ],
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(results: list[dict]) -> dict:
    scored = [
        result
        for result in results
        if result["recall"] is not None
    ]

    by_type = defaultdict(list)

    for result in scored:
        by_type[result["type"]].append(result)

    def metrics(items: list[dict]) -> dict:
        return {
            "queries": len(items),
            f"recall@{TOP_K}": round(
                mean([item["recall"] for item in items]),
                3,
            ),
            "mrr": round(
                mean([item["reciprocal_rank"] for item in items]),
                3,
            ),
            "hit_rate": round(
                mean([
                    1.0 if item["found"] else 0.0
                    for item in items
                ]),
                3,
            ),
        }

    total_expected = sum(
        result["expected_count"]
        + result["outside_candidate_pool_count"]
        + result["unresolved_count"]
        for result in results
    )
    eligible_expected = sum(
        result["expected_count"] for result in results
    )
    outside_pool_expected = sum(
        result["outside_candidate_pool_count"] for result in results
    )
    unresolved_expected = sum(
        result["unresolved_count"] for result in results
    )

    return {
        "overall": metrics(scored),
        "candidate_pool_coverage": {
            "expected_references": total_expected,
            "eligible_in_english_pool": eligible_expected,
            "outside_english_pool": outside_pool_expected,
            "unresolved_in_database": unresolved_expected,
            "coverage": round(
                eligible_expected / total_expected, 3
            ) if total_expected else None,
        },
        "by_type": {
            query_type: metrics(items)
            for query_type, items in sorted(by_type.items())
        },
        "unscored_queries": [
            result["id"]
            for result in results
            if result["recall"] is None
        ],
        "expected_outside_candidate_pool": [
            {
                "query_id": result["id"],
                "books": result["outside_candidate_pool"],
            }
            for result in results
            if result["outside_candidate_pool"]
        ],
    }


def print_report(
    model_name: str,
    results: list[dict],
    summary: dict,
) -> None:
    print()
    print("=" * 100)
    print(f"MODEL: {model_name}")
    print("=" * 100)

    print(
        f"{'id':<5} {'type':<11} {'recall':>6} "
        f"{'rank':>5}  top result"
    )
    print("-" * 100)

    for result in results:
        recall = (
            "-"
            if result["recall"] is None
            else f"{result['recall']:.2f}"
        )

        rank = result["first_rank"] or "miss"

        top = (
            result["top_results"][0]
            if result["top_results"]
            else None
        )

        top_text = (
            f"{top['title'][:50]} ({top['author']})"
            if top
            else "(no results)"
        )

        print(
            f"{result['id']:<5} "
            f"{result['type']:<11} "
            f"{recall:>6} "
            f"{rank:>5}  "
            f"{top_text}"
        )

    print()
    print("Overall:", summary["overall"])
    print("Candidate-pool coverage:", summary["candidate_pool_coverage"])

    for query_type, metrics in summary["by_type"].items():
        print(f"  {query_type:<11}", metrics)

    unresolved = [
        (result["id"], expected)
        for result in results
        for expected in result["unresolved"]
    ]

    if unresolved:
        print("\nExpected books not found in the database:")

        for query_id, expected in unresolved:
            print(
                f"  {query_id}: "
                f"{expected['title']} / {expected['author']}"
            )

    outside = [
        (result["id"], expected)
        for result in results
        for expected in result["outside_candidate_pool"]
    ]

    if outside:
        print("\nExpected books outside the English candidate pool:")

        for query_id, expected in outside:
            print(
                f"  {query_id}: "
                f"{expected['title']} / {expected['author']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MiniLM and BGE-small retrieval."
    )

    parser.add_argument(
        "--min-ratings",
        type=int,
        default=None,
        help="Only return books with at least this many ratings.",
    )

    parser.add_argument(
        "--mode",
        choices=["semantic", "hybrid"],
        default="semantic",
        help="semantic or hybrid retrieval.",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=SEMANTIC_WEIGHT,
        help="Semantic RRF weight for hybrid mode (default matches API: 2.0).",
    )
    parser.add_argument(
        "--keyword-weight",
        type=float,
        default=KEYWORD_WEIGHT,
        help="Keyword RRF weight for hybrid mode (default matches API: 1.0).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.semantic_weight < 0 or args.keyword_weight < 0:
        raise ValueError("Hybrid weights must be non-negative")
    if args.semantic_weight == 0 and args.keyword_weight == 0:
        raise ValueError("At least one hybrid weight must be greater than zero")

    queries = json.loads(
        QUERIES_PATH.read_text()
    )["queries"]

    logger.info("Loaded %d evaluation queries", len(queries))

    with Session(engine) as session:
        candidate_ids = fetch_english_candidate_ids(session)

        candidate_id_set = set(candidate_ids)

        logger.info(
            "Using identical candidate pool of %d English books",
            len(candidate_ids),
        )

        expected_total = sum(
            len(query["expected"])
            for query in queries
        )

        logger.info("Expected book references: %d", expected_total)

        model_configs = [
            {
                "name": MINILM_MODEL,
                "table": MINILM_TABLE,
            },
            {
                "name": BGE_MODEL,
                "table": BGE_TABLE,
            },
        ]

        all_results = {}
        all_summaries = {}

        for config in model_configs:
            model_name = config["name"]

            logger.info(
                "Evaluating %s | mode=%s | min_ratings=%s | "
                "hybrid_weights=%s:%s",
                model_name,
                args.mode,
                args.min_ratings,
                args.semantic_weight,
                args.keyword_weight,
            )

            model = load_model(model_name)

            results = []

            for query in queries:
                logger.info(
                    "%s | %s: %s",
                    model_name,
                    query["id"],
                    query["query"],
                )

                result = evaluate_query(
                    session=session,
                    model=model,
                    model_name=model_name,
                    table_name=config["table"],
                    query=query,
                    candidate_ids=candidate_ids,
                    candidate_id_set=candidate_id_set,
                    min_ratings=args.min_ratings,
                    mode=args.mode,
                    semantic_weight=args.semantic_weight,
                    keyword_weight=args.keyword_weight,
                )

                results.append(result)

            summary = summarize(results)

            print_report(
                model_name,
                results,
                summary,
            )

            all_results[model_name] = results
            all_summaries[model_name] = summary

            del model
            gc.collect()

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print("\n" + "=" * 100)
    print("FINAL MODEL COMPARISON")
    print("=" * 100)

    for model_name, summary in all_summaries.items():
        print(f"\n{model_name}")
        print(summary["overall"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    filter_label = (
        f"min{args.min_ratings}"
        if args.min_ratings is not None
        else "nofilter"
    )

    output_path = (
        RESULTS_DIR
        / (
            f"retrieval_comparison_"
            f"{datetime.now():%Y%m%d_%H%M%S}_"
            f"{args.mode}_{filter_label}.json"
        )
    )

    output_path.write_text(
        json.dumps(
            {
                "models": list(all_results.keys()),
                "candidate_pool": {
                    "language": "English",
                    "count": len(candidate_ids),
                    "definition": (
                        "Books detected as English with embeddings present "
                        "for both MiniLM and BGE."
                    ),
                    "models": {
                        "minilm": MINILM_MODEL,
                        "bge": BGE_MODEL,
                    },
                },
                "top_k": TOP_K,
                "mode": args.mode,
                "hybrid_weights": {
                    "semantic": args.semantic_weight,
                    "keyword": args.keyword_weight,
                },
                "min_ratings": args.min_ratings,
                "summaries": all_summaries,
                "results": all_results,
            },
            indent=2,
        )
    )

    logger.info(
        "Saved comparison results to %s",
        output_path.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()
