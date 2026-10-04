import sqlite3
import unittest

import database


class DatabaseBatchTests(unittest.TestCase):
    def test_multi_value_insert_returns_aligned_ids_across_chunks(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT)")
        rows = [(f"value-{index}",) for index in range(205)]

        ids = database._insert_many_returning_ids(
            conn,
            "INSERT INTO items (value)",
            "(?)",
            rows,
            chunk_size=50,
        )

        self.assertEqual(ids, list(range(1, 206)))
        stored = conn.execute("SELECT value FROM items ORDER BY id").fetchall()
        self.assertEqual([row[0] for row in stored], [row[0] for row in rows])


if __name__ == "__main__":
    unittest.main()
