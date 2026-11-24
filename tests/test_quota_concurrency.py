from urllib.parse import urlparse
import os
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import psycopg


DATABASE_URL = os.environ["TEST_DATABASE_URL"]

parsed_url = urlparse(DATABASE_URL)

if (
    parsed_url.scheme != "postgresql"
    or parsed_url.hostname != "localhost"
    or parsed_url.port != 5432
    or parsed_url.path != "/documind_test"
    or parsed_url.username != "quota_test"
    or parsed_url.query
):
    raise RuntimeError(
        "Safety check failed: tests must use the isolated "
        "DocuMind test database."
    )


def send_request(visitor_id, barrier):
    with psycopg.connect(DATABASE_URL, connect_timeout=10) as conn:
        barrier.wait(timeout=30)

        result = conn.execute(
            "SELECT public.use_ai_request(%s, %s, %s)",
            (visitor_id, 10, 100),
        ).fetchone()

        return result[0]


def run_concurrent_requests(visitor_ids):
    barrier = Barrier(len(visitor_ids))

    with ThreadPoolExecutor(max_workers=len(visitor_ids)) as executor:
        futures = [
            executor.submit(send_request, visitor_id, barrier)
            for visitor_id in visitor_ids
        ]

        return Counter(future.result() for future in futures)


class TestQuotaConcurrency(unittest.TestCase):

    def setUp(self):
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute(
                "TRUNCATE public.daily_usage, public.global_daily_usage"
            )

    def get_global_count(self):
        with psycopg.connect(DATABASE_URL) as conn:
            return conn.execute(
                """
                SELECT COALESCE(SUM(request_count), 0)
                FROM public.global_daily_usage
                """
            ).fetchone()[0]

    def get_visitor_count(self, visitor_id):
        with psycopg.connect(DATABASE_URL) as conn:
            return conn.execute(
                """
                SELECT COALESCE(SUM(question_count), 0)
                FROM public.daily_usage
                WHERE visitor_id = %s
                """,
                (visitor_id,),
            ).fetchone()[0]

    def test_visitor_quota_concurrency(self):
        visitor_ids = ["test-visitor"] * 20

        results = run_concurrent_requests(visitor_ids)

        self.assertEqual(results["allowed"], 10)
        self.assertEqual(results["visitor_limit"], 10)
        self.assertEqual(self.get_global_count(), 10)
        self.assertEqual(self.get_visitor_count("test-visitor"), 10)

    def test_global_quota_concurrency(self):
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute(
                """
                INSERT INTO public.global_daily_usage
                    (usage_date, request_count)
                VALUES (
                    (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::date,
                    95
                )
                """
            )

        visitor_ids = [
            f"test-visitor-{i}" for i in range(20)
        ]

        results = run_concurrent_requests(visitor_ids)

        self.assertEqual(results["allowed"], 5)
        self.assertEqual(results["global_limit"], 15)
        self.assertEqual(self.get_global_count(), 100)

        total_visitor_count = sum(
            self.get_visitor_count(visitor_id)
            for visitor_id in visitor_ids
        )

        self.assertEqual(total_visitor_count, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
