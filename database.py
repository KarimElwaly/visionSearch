"""SQLite database for video/frame/detection metadata with FTS5 search."""
import sqlite3
import threading
from contextlib import contextmanager
from config import DB_PATH


_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """Get a thread-local database connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


@contextmanager
def get_db():
    """Context manager for database operations."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db():
    """Initialize database schema."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                duration REAL,
                fps REAL,
                total_frames INTEGER,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending'
            );
            
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                frame_path TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
            );
            
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL,
                faiss_id INTEGER,
                person_faiss_id INTEGER,
                class_name TEXT,
                confidence REAL,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                crop_path TEXT,
                caption TEXT,
                FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE
            );
            
            -- Face crops linked to detections (person → face)
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER NOT NULL,
                frame_id INTEGER NOT NULL,
                face_faiss_id INTEGER,
                confidence REAL,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                crop_path TEXT,
                FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE,
                FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE
            );
            
            CREATE VIRTUAL TABLE IF NOT EXISTS detections_fts USING fts5(
                caption,
                content='detections',
                content_rowid='id'
            );
            
            CREATE TRIGGER IF NOT EXISTS detections_ai AFTER INSERT ON detections BEGIN
                INSERT INTO detections_fts(rowid, caption) VALUES (new.id, new.caption);
            END;
            
            CREATE TRIGGER IF NOT EXISTS detections_au AFTER UPDATE ON detections BEGIN
                INSERT INTO detections_fts(detections_fts, rowid, caption) VALUES('delete', old.id, old.caption);
                INSERT INTO detections_fts(rowid, caption) VALUES (new.id, new.caption);
            END;
            
            CREATE TRIGGER IF NOT EXISTS detections_ad AFTER DELETE ON detections BEGIN
                INSERT INTO detections_fts(detections_fts, rowid, caption) VALUES('delete', old.id, old.caption);
            END;
            
            -- Full-frame entries (no bounding box, scene-level embeddings)
            CREATE TABLE IF NOT EXISTS frame_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL UNIQUE,
                faiss_id INTEGER,
                caption TEXT,
                FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE
            );
            
            CREATE VIRTUAL TABLE IF NOT EXISTS frame_embeddings_fts USING fts5(
                caption,
                content='frame_embeddings',
                content_rowid='id'
            );
            
            CREATE TRIGGER IF NOT EXISTS frame_emb_ai AFTER INSERT ON frame_embeddings BEGIN
                INSERT INTO frame_embeddings_fts(rowid, caption) VALUES (new.id, new.caption);
            END;
            CREATE TRIGGER IF NOT EXISTS frame_emb_au AFTER UPDATE ON frame_embeddings BEGIN
                INSERT INTO frame_embeddings_fts(frame_embeddings_fts, rowid, caption) VALUES('delete', old.id, old.caption);
                INSERT INTO frame_embeddings_fts(rowid, caption) VALUES (new.id, new.caption);
            END;
        """)

        # --- Migrations for existing databases ---
        try:
            conn.execute("ALTER TABLE detections ADD COLUMN person_faiss_id INTEGER")
            print("[db] Migrated: added person_faiss_id to detections")
        except Exception:
            pass  # Column already exists

        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER NOT NULL,
                    frame_id INTEGER NOT NULL,
                    face_faiss_id INTEGER,
                    confidence REAL,
                    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                    crop_path TEXT,
                    FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE,
                    FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE
                );
            """)
        except Exception:
            pass


# ---- Video CRUD ----

def add_video(filename: str, path: str, duration: float, fps: float, total_frames: int) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO videos (filename, path, duration, fps, total_frames, status) VALUES (?, ?, ?, ?, ?, 'indexing')",
            (filename, path, duration, fps, total_frames)
        )
        return cur.lastrowid


def update_video_status(video_id: int, status: str):
    with get_db() as conn:
        conn.execute("UPDATE videos SET status = ? WHERE id = ?", (status, video_id))


def get_video_by_path(path: str):
    with get_db() as conn:
        return conn.execute("SELECT * FROM videos WHERE path = ?", (path,)).fetchone()


def get_all_videos():
    with get_db() as conn:
        return conn.execute("SELECT * FROM videos ORDER BY indexed_at DESC").fetchall()


def delete_video(video_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))


# ---- Frame CRUD ----

def add_frame(video_id: int, frame_number: int, timestamp: float, frame_path: str) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO frames (video_id, frame_number, timestamp, frame_path) VALUES (?, ?, ?, ?)",
            (video_id, frame_number, timestamp, frame_path)
        )
        return cur.lastrowid


# ---- Detection CRUD ----

def add_detection(frame_id: int, class_name: str, confidence: float,
                  bbox: tuple, crop_path: str, faiss_id: int = None, caption: str = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO detections 
               (frame_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path, faiss_id, caption) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (frame_id, class_name, confidence, *bbox, crop_path, faiss_id, caption)
        )
        return cur.lastrowid


def update_detection_caption(detection_id: int, caption: str):
    with get_db() as conn:
        conn.execute("UPDATE detections SET caption = ? WHERE id = ?", (caption, detection_id))


def add_frame_embedding(frame_id: int, faiss_id: int, caption: str = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO frame_embeddings (frame_id, faiss_id, caption) VALUES (?, ?, ?)",
            (frame_id, faiss_id, caption)
        )
        return cur.lastrowid


# ---- Batch Inserts (single-transaction for speed) ----

def batch_add_frames(rows: list) -> list:
    """Insert many frames in one transaction. Each row: (video_id, frame_number, timestamp, frame_path).
    Returns list of rowids in insertion order."""
    with get_db() as conn:
        ids = []
        for r in rows:
            cur = conn.execute(
                "INSERT INTO frames (video_id, frame_number, timestamp, frame_path) VALUES (?, ?, ?, ?)", r
            )
            ids.append(cur.lastrowid)
        return ids


def batch_add_frame_embeddings(rows: list):
    """Insert many frame_embeddings in one transaction. Each row: (frame_id, faiss_id, caption)."""
    with get_db() as conn:
        conn.executemany(
            "INSERT INTO frame_embeddings (frame_id, faiss_id, caption) VALUES (?, ?, ?)", rows
        )


def batch_add_detections(rows: list):
    """Insert many detections in one transaction.
    Each row: (frame_id, class_name, confidence, x1, y1, x2, y2, crop_path, faiss_id, caption).
    Returns list of rowids."""
    with get_db() as conn:
        ids = []
        for r in rows:
            cur = conn.execute(
                """INSERT INTO detections
                   (frame_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path, faiss_id, caption)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", r
            )
            ids.append(cur.lastrowid)
        return ids


def update_detection_person_faiss_id(detection_id: int, person_faiss_id: int):
    """Set the person FAISS ID for a detection."""
    with get_db() as conn:
        conn.execute("UPDATE detections SET person_faiss_id = ? WHERE id = ?", (person_faiss_id, detection_id))


def batch_update_person_faiss_ids(rows: list):
    """Update person_faiss_id for many detections. Each row: (person_faiss_id, detection_id)."""
    with get_db() as conn:
        conn.executemany("UPDATE detections SET person_faiss_id = ? WHERE id = ?", rows)


def batch_add_faces(rows: list):
    """Insert many face records in one transaction.
    Each row: (detection_id, frame_id, face_faiss_id, confidence, x1, y1, x2, y2, crop_path)."""
    with get_db() as conn:
        conn.executemany(
            """INSERT INTO faces
               (detection_id, frame_id, face_faiss_id, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows
        )


# ---- Search ----

def search_by_text(query: str, limit: int = 20):
    """Full-text search on detection captions."""
    with get_db() as conn:
        # Search detections
        results = conn.execute("""
            SELECT d.*, f.frame_path, f.timestamp, f.frame_number, 
                   v.filename as video_filename, v.path as video_path,
                   rank
            FROM detections_fts 
            JOIN detections d ON d.id = detections_fts.rowid
            JOIN frames f ON f.id = d.frame_id
            JOIN videos v ON v.id = f.video_id
            WHERE detections_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit)).fetchall()
        return results


def search_frames_by_text(query: str, limit: int = 20):
    """Full-text search on frame-level captions."""
    with get_db() as conn:
        results = conn.execute("""
            SELECT fe.*, f.frame_path, f.timestamp, f.frame_number,
                   v.filename as video_filename, v.path as video_path,
                   rank
            FROM frame_embeddings_fts
            JOIN frame_embeddings fe ON fe.id = frame_embeddings_fts.rowid
            JOIN frames f ON f.id = fe.frame_id
            JOIN videos v ON v.id = f.video_id
            WHERE frame_embeddings_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit)).fetchall()
        return results


def get_detections_by_faiss_ids(faiss_ids: list):
    """Get detections by their FAISS index IDs."""
    placeholders = ",".join("?" * len(faiss_ids))
    with get_db() as conn:
        return conn.execute(f"""
            SELECT d.*, f.frame_path, f.timestamp, f.frame_number,
                   v.filename as video_filename, v.path as video_path
            FROM detections d
            JOIN frames f ON f.id = d.frame_id
            JOIN videos v ON v.id = f.video_id
            WHERE d.faiss_id IN ({placeholders})
        """, faiss_ids).fetchall()


def get_frame_embeddings_by_faiss_ids(faiss_ids: list):
    """Get frame embeddings by their FAISS index IDs."""
    placeholders = ",".join("?" * len(faiss_ids))
    with get_db() as conn:
        return conn.execute(f"""
            SELECT fe.*, f.frame_path, f.timestamp, f.frame_number,
                   v.filename as video_filename, v.path as video_path
            FROM frame_embeddings fe
            JOIN frames f ON f.id = fe.frame_id
            JOIN videos v ON v.id = f.video_id
            WHERE fe.faiss_id IN ({placeholders})
        """, faiss_ids).fetchall()


def get_persons_by_faiss_ids(faiss_ids: list):
    """Get person detections by their person FAISS index IDs."""
    placeholders = ",".join("?" * len(faiss_ids))
    with get_db() as conn:
        return conn.execute(f"""
            SELECT d.*, f.frame_path, f.timestamp, f.frame_number,
                   v.filename as video_filename, v.path as video_path
            FROM detections d
            JOIN frames f ON f.id = d.frame_id
            JOIN videos v ON v.id = f.video_id
            WHERE d.person_faiss_id IN ({placeholders})
        """, faiss_ids).fetchall()


def get_faces_by_faiss_ids(faiss_ids: list):
    """Get face records by their face FAISS index IDs."""
    placeholders = ",".join("?" * len(faiss_ids))
    with get_db() as conn:
        return conn.execute(f"""
            SELECT fc.*, d.crop_path as person_crop_path, d.class_name,
                   f.frame_path, f.timestamp, f.frame_number,
                   v.filename as video_filename, v.path as video_path
            FROM faces fc
            JOIN detections d ON d.id = fc.detection_id
            JOIN frames f ON f.id = fc.frame_id
            JOIN videos v ON v.id = f.video_id
            WHERE fc.face_faiss_id IN ({placeholders})
        """, faiss_ids).fetchall()

