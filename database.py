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
        _local.conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
        _local.conn.execute("PRAGMA temp_store=MEMORY")
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
                camera_id TEXT,
                location TEXT,
                recording_start_time TEXT,
                timezone TEXT,
                index_profile TEXT,
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
                track_id INTEGER,
                faiss_id INTEGER,
                person_faiss_id INTEGER,
                class_name TEXT,
                confidence REAL,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                crop_path TEXT,
                caption TEXT,
                FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE
            );

            -- CCTV event tracks: groups repeated frame detections into one moving object/person.
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                local_track_id INTEGER NOT NULL,
                class_name TEXT,
                start_frame_id INTEGER,
                end_frame_id INTEGER,
                start_frame_number INTEGER,
                end_frame_number INTEGER,
                start_timestamp REAL,
                end_timestamp REAL,
                detection_count INTEGER DEFAULT 0,
                best_detection_id INTEGER,
                best_confidence REAL DEFAULT 0,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
                FOREIGN KEY (start_frame_id) REFERENCES frames(id) ON DELETE SET NULL,
                FOREIGN KEY (end_frame_id) REFERENCES frames(id) ON DELETE SET NULL,
                FOREIGN KEY (best_detection_id) REFERENCES detections(id) ON DELETE SET NULL,
                UNIQUE(video_id, local_track_id)
            );

            CREATE TABLE IF NOT EXISTS track_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                detection_id INTEGER NOT NULL UNIQUE,
                frame_id INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                confidence REAL,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
                FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE,
                FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS track_attributes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
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
            conn.execute("ALTER TABLE detections ADD COLUMN track_id INTEGER")
            print("[db] Migrated: added track_id to detections")
        except Exception:
            pass  # Column already exists

        # Resume-support columns on videos (added in Phase 3 of scalability upgrade).
        for column, ddl in [
            ("last_frame_number", "ALTER TABLE videos ADD COLUMN last_frame_number INTEGER DEFAULT 0"),
            ("frames_processed", "ALTER TABLE videos ADD COLUMN frames_processed INTEGER DEFAULT 0"),
            ("error_message", "ALTER TABLE videos ADD COLUMN error_message TEXT"),
            ("camera_id", "ALTER TABLE videos ADD COLUMN camera_id TEXT"),
            ("location", "ALTER TABLE videos ADD COLUMN location TEXT"),
            ("recording_start_time", "ALTER TABLE videos ADD COLUMN recording_start_time TEXT"),
            ("timezone", "ALTER TABLE videos ADD COLUMN timezone TEXT"),
            ("index_profile", "ALTER TABLE videos ADD COLUMN index_profile TEXT"),
        ]:
            try:
                conn.execute(ddl)
                print(f"[db] Migrated: added {column} to videos")
            except Exception:
                pass

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

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                local_track_id INTEGER NOT NULL,
                class_name TEXT,
                start_frame_id INTEGER,
                end_frame_id INTEGER,
                start_frame_number INTEGER,
                end_frame_number INTEGER,
                start_timestamp REAL,
                end_timestamp REAL,
                detection_count INTEGER DEFAULT 0,
                best_detection_id INTEGER,
                best_confidence REAL DEFAULT 0,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
                FOREIGN KEY (start_frame_id) REFERENCES frames(id) ON DELETE SET NULL,
                FOREIGN KEY (end_frame_id) REFERENCES frames(id) ON DELETE SET NULL,
                FOREIGN KEY (best_detection_id) REFERENCES detections(id) ON DELETE SET NULL,
                UNIQUE(video_id, local_track_id)
            );

            CREATE TABLE IF NOT EXISTS track_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                detection_id INTEGER NOT NULL UNIQUE,
                frame_id INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                confidence REAL,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
                FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE,
                FOREIGN KEY (frame_id) REFERENCES frames(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS track_attributes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
            );
        """)

        # --- Performance indexes (critical at scale) ---
        # SQLite silently no-ops CREATE INDEX IF NOT EXISTS, so this is safe on repeat runs.
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_frames_video_id ON frames(video_id);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_frames_video_frameno ON frames(video_id, frame_number);
            CREATE INDEX IF NOT EXISTS idx_detections_frame_id ON detections(frame_id);
            CREATE INDEX IF NOT EXISTS idx_detections_track_id ON detections(track_id);
            CREATE INDEX IF NOT EXISTS idx_detections_faiss_id ON detections(faiss_id);
            CREATE INDEX IF NOT EXISTS idx_detections_person_faiss_id ON detections(person_faiss_id);
            CREATE INDEX IF NOT EXISTS idx_faces_face_faiss_id ON faces(face_faiss_id);
            CREATE INDEX IF NOT EXISTS idx_faces_detection_id ON faces(detection_id);
            CREATE INDEX IF NOT EXISTS idx_frame_embeddings_faiss_id ON frame_embeddings(faiss_id);
            CREATE INDEX IF NOT EXISTS idx_videos_camera_id ON videos(camera_id);
            CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
            CREATE INDEX IF NOT EXISTS idx_tracks_video_class ON tracks(video_id, class_name);
            CREATE INDEX IF NOT EXISTS idx_tracks_video_time ON tracks(video_id, start_timestamp, end_timestamp);
            CREATE INDEX IF NOT EXISTS idx_tracks_best_detection ON tracks(best_detection_id);
            CREATE INDEX IF NOT EXISTS idx_track_detections_track_id ON track_detections(track_id);
            CREATE INDEX IF NOT EXISTS idx_track_detections_frame_id ON track_detections(frame_id);
            CREATE INDEX IF NOT EXISTS idx_track_detections_timestamp ON track_detections(timestamp);
        """)


def vacuum_analyze():
    """Reclaim space and refresh planner stats. Call after bulk deletes."""
    with get_db() as conn:
        # VACUUM cannot run inside a transaction; commit first, then run outside.
        conn.commit()
    conn = get_connection()
    try:
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
    except Exception as e:
        print(f"[db] vacuum_analyze failed: {e}")


def update_video_checkpoint(video_id: int, last_frame_number: int, frames_processed: int):
    """Persist per-video checkpoint so indexing can resume after a crash."""
    with get_db() as conn:
        conn.execute(
            "UPDATE videos SET last_frame_number = ?, frames_processed = ? WHERE id = ?",
            (last_frame_number, frames_processed, video_id)
        )


def set_video_error(video_id: int, error_message: str):
    """Mark a video as ``failed`` with the exception message. Called from the
    indexer's outer try/except so crashes don't leave rows stuck in 'indexing'."""
    with get_db() as conn:
        conn.execute(
            "UPDATE videos SET status = 'failed', error_message = ? WHERE id = ?",
            (error_message[:2000] if error_message else None, video_id),
        )


def get_max_frame_number(video_id: int) -> int:
    """Return the largest ``frame_number`` already stored for ``video_id`` (0 if none)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT MAX(frame_number) AS m FROM frames WHERE video_id = ?",
            (video_id,)
        ).fetchone()
    return int(row["m"]) if row and row["m"] is not None else 0


# ---- Video CRUD ----

def add_video(filename: str, path: str, duration: float, fps: float, total_frames: int,
              camera_id: str = None, location: str = None, recording_start_time: str = None,
              timezone: str = None, index_profile: str = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO videos
               (filename, path, duration, fps, total_frames, camera_id, location,
                recording_start_time, timezone, index_profile, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'indexing')""",
            (filename, path, duration, fps, total_frames, camera_id, location,
             recording_start_time, timezone, index_profile)
        )
        return cur.lastrowid


def update_video_metadata(video_id: int, camera_id: str = None, location: str = None,
                          recording_start_time: str = None, timezone: str = None,
                          index_profile: str = None):
    """Update optional CCTV metadata without overwriting fields with empty values."""
    updates = []
    params = []
    for column, value in [
        ("camera_id", camera_id),
        ("location", location),
        ("recording_start_time", recording_start_time),
        ("timezone", timezone),
        ("index_profile", index_profile),
    ]:
        if value:
            updates.append(f"{column} = ?")
            params.append(value)
    if not updates:
        return
    params.append(video_id)
    with get_db() as conn:
        conn.execute(f"UPDATE videos SET {', '.join(updates)} WHERE id = ?", params)


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


def get_faiss_ids_for_video(video_id: int) -> dict:
    """Return all FAISS ids referenced by this video, grouped by index.

    Used by the cascade-delete path to actually remove vectors from FAISS
    (otherwise deleted content would leak into future searches forever).
    """
    with get_db() as conn:
        det_rows = conn.execute(
            """SELECT d.faiss_id, d.person_faiss_id
               FROM detections d
               JOIN frames f ON f.id = d.frame_id
               WHERE f.video_id = ?""",
            (video_id,)
        ).fetchall()
        frame_rows = conn.execute(
            """SELECT fe.faiss_id
               FROM frame_embeddings fe
               JOIN frames f ON f.id = fe.frame_id
               WHERE f.video_id = ?""",
            (video_id,)
        ).fetchall()
        face_rows = conn.execute(
            """SELECT fc.face_faiss_id
               FROM faces fc
               JOIN frames f ON f.id = fc.frame_id
               WHERE f.video_id = ?""",
            (video_id,)
        ).fetchall()

    general = [r["faiss_id"] for r in det_rows if r["faiss_id"] is not None]
    general += [r["faiss_id"] for r in frame_rows if r["faiss_id"] is not None]
    person = [r["person_faiss_id"] for r in det_rows if r["person_faiss_id"] is not None]
    face = [r["face_faiss_id"] for r in face_rows if r["face_faiss_id"] is not None]
    return {"general": general, "person": person, "face": face}


def get_storage_stats() -> dict:
    """Aggregate per-video counts (rows in each table) for the storage endpoint."""
    with get_db() as conn:
        videos = conn.execute(
            """SELECT id, filename, path, duration, status, camera_id, location,
                      recording_start_time, timezone, index_profile
               FROM videos"""
        ).fetchall()
        rows = []
        for v in videos:
            vid = v["id"]
            frames_n = conn.execute("SELECT COUNT(*) AS n FROM frames WHERE video_id = ?", (vid,)).fetchone()["n"]
            det_n = conn.execute(
                "SELECT COUNT(*) AS n FROM detections d JOIN frames f ON f.id = d.frame_id WHERE f.video_id = ?",
                (vid,)
            ).fetchone()["n"]
            face_n = conn.execute(
                "SELECT COUNT(*) AS n FROM faces fc JOIN frames f ON f.id = fc.frame_id WHERE f.video_id = ?",
                (vid,)
            ).fetchone()["n"]
            track_n = conn.execute("SELECT COUNT(*) AS n FROM tracks WHERE video_id = ?", (vid,)).fetchone()["n"]
            rows.append({
                "id": vid, "filename": v["filename"], "path": v["path"],
                "duration": v["duration"], "status": v["status"],
                "camera_id": v["camera_id"], "location": v["location"],
                "recording_start_time": v["recording_start_time"], "timezone": v["timezone"],
                "index_profile": v["index_profile"],
                "frames": frames_n, "detections": det_n, "faces": face_n, "tracks": track_n,
            })
    return {"videos": rows, "tracks": get_track_stats()}


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


def batch_link_track_detections(video_id: int, observations: list):
    """Persist tracker assignments for a batch of detections.

    Each observation is a dict with: local_track_id, class_name, frame_id,
    detection_id, frame_number, timestamp, confidence, bbox.
    """
    if not observations:
        return

    with get_db() as conn:
        track_id_cache = {}
        for obs in observations:
            local_track_id = int(obs["local_track_id"])
            class_name = obs.get("class_name") or "object"
            frame_id = int(obs["frame_id"])
            detection_id = int(obs["detection_id"])
            frame_number = int(obs["frame_number"])
            timestamp = float(obs["timestamp"])
            confidence = float(obs.get("confidence") or 0.0)
            x1, y1, x2, y2 = obs["bbox"]

            conn.execute(
                """INSERT INTO tracks
                   (video_id, local_track_id, class_name, start_frame_id, end_frame_id,
                    start_frame_number, end_frame_number, start_timestamp, end_timestamp,
                    detection_count, best_detection_id, best_confidence, status, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 'active', CURRENT_TIMESTAMP)
                   ON CONFLICT(video_id, local_track_id) DO UPDATE SET
                       end_frame_id = excluded.end_frame_id,
                       end_frame_number = excluded.end_frame_number,
                       end_timestamp = excluded.end_timestamp,
                       detection_count = tracks.detection_count + 1,
                       best_detection_id = CASE
                           WHEN excluded.best_confidence > COALESCE(tracks.best_confidence, 0)
                           THEN excluded.best_detection_id ELSE tracks.best_detection_id END,
                       best_confidence = CASE
                           WHEN excluded.best_confidence > COALESCE(tracks.best_confidence, 0)
                           THEN excluded.best_confidence ELSE tracks.best_confidence END,
                       status = 'active',
                       updated_at = CURRENT_TIMESTAMP""",
                (video_id, local_track_id, class_name, frame_id, frame_id, frame_number,
                 frame_number, timestamp, timestamp, detection_id, confidence)
            )

            cache_key = (video_id, local_track_id)
            if cache_key not in track_id_cache:
                row = conn.execute(
                    "SELECT id FROM tracks WHERE video_id = ? AND local_track_id = ?",
                    (video_id, local_track_id)
                ).fetchone()
                if row is None:
                    continue
                track_id_cache[cache_key] = int(row["id"])
            track_id = track_id_cache[cache_key]

            conn.execute(
                """INSERT OR IGNORE INTO track_detections
                   (track_id, detection_id, frame_id, frame_number, timestamp, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (track_id, detection_id, frame_id, frame_number, timestamp, confidence,
                 x1, y1, x2, y2)
            )
            conn.execute("UPDATE detections SET track_id = ? WHERE id = ?", (track_id, detection_id))


def get_next_track_local_id(video_id: int) -> int:
    """Return the next local track id for a video (used when resuming indexing)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(local_track_id), 0) + 1 AS next_id FROM tracks WHERE video_id = ?",
            (video_id,)
        ).fetchone()
    return int(row["next_id"] or 1)


def get_track_stats(video_id: int = None) -> dict:
    """Return aggregate track counts, optionally scoped to one video."""
    where = "WHERE video_id = ?" if video_id is not None else ""
    params = (video_id,) if video_id is not None else ()
    with get_db() as conn:
        summary = conn.execute(
            f"""SELECT COUNT(*) AS tracks,
                       COALESCE(SUM(detection_count), 0) AS linked_detections,
                       COALESCE(MAX(end_timestamp), 0) AS latest_timestamp
                FROM tracks {where}""",
            params
        ).fetchone()
        by_class = conn.execute(
            f"""SELECT class_name, COUNT(*) AS tracks,
                       COALESCE(SUM(detection_count), 0) AS detections
                FROM tracks {where}
                GROUP BY class_name
                ORDER BY tracks DESC""",
            params
        ).fetchall()
    return {
        "tracks": int(summary["tracks"] or 0),
        "linked_detections": int(summary["linked_detections"] or 0),
        "latest_timestamp": float(summary["latest_timestamp"] or 0),
        "by_class": [dict(r) for r in by_class],
    }


def get_tracks_for_video(video_id: int, class_name: str = None, limit: int = 100):
    """Return representative tracks for one video."""
    params = [video_id]
    class_filter = ""
    if class_name:
        class_filter = "AND t.class_name = ?"
        params.append(class_name)
    params.append(limit)

    with get_db() as conn:
        return conn.execute(f"""
            SELECT t.*, d.crop_path, d.confidence AS best_detection_confidence,
                   d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2,
                   f.frame_path, f.timestamp AS best_timestamp, f.frame_number AS best_frame_number,
                   v.filename AS video_filename, v.path AS video_path
            FROM tracks t
            JOIN videos v ON v.id = t.video_id
            LEFT JOIN detections d ON d.id = t.best_detection_id
            LEFT JOIN frames f ON f.id = d.frame_id
            WHERE t.video_id = ? {class_filter}
            ORDER BY t.detection_count DESC, t.best_confidence DESC, t.start_timestamp ASC
            LIMIT ?
        """, params).fetchall()


def get_track_timeline(track_id: int, limit: int = 500):
    """Return ordered detections for a track."""
    with get_db() as conn:
        return conn.execute("""
            SELECT td.*, d.crop_path, d.class_name, f.frame_path,
                   v.filename AS video_filename, v.path AS video_path
            FROM track_detections td
            JOIN tracks t ON t.id = td.track_id
            JOIN detections d ON d.id = td.detection_id
            JOIN frames f ON f.id = td.frame_id
            JOIN videos v ON v.id = t.video_id
            WHERE td.track_id = ?
            ORDER BY td.timestamp ASC
            LIMIT ?
        """, (track_id, limit)).fetchall()


# ---- Search ----

def search_by_text(query: str, limit: int = 20):
    """Full-text search on detection captions."""
    with get_db() as conn:
        # Search detections
        results = conn.execute("""
            SELECT d.*, f.frame_path, f.timestamp, f.frame_number, 
                 v.id as video_id, v.filename as video_filename, v.path as video_path,
                 v.camera_id, v.location, v.recording_start_time, v.timezone,
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
                 v.id as video_id, v.filename as video_filename, v.path as video_path,
                 v.camera_id, v.location, v.recording_start_time, v.timezone,
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
                 v.id as video_id, v.filename as video_filename, v.path as video_path,
                 v.camera_id, v.location, v.recording_start_time, v.timezone
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
                 v.id as video_id, v.filename as video_filename, v.path as video_path,
                 v.camera_id, v.location, v.recording_start_time, v.timezone
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
                 v.id as video_id, v.filename as video_filename, v.path as video_path,
                 v.camera_id, v.location, v.recording_start_time, v.timezone
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
            SELECT fc.*, d.crop_path as person_crop_path, d.class_name, d.track_id,
                   f.frame_path, f.timestamp, f.frame_number,
                 v.id as video_id, v.filename as video_filename, v.path as video_path,
                 v.camera_id, v.location, v.recording_start_time, v.timezone
            FROM faces fc
            JOIN detections d ON d.id = fc.detection_id
            JOIN frames f ON f.id = fc.frame_id
            JOIN videos v ON v.id = f.video_id
            WHERE fc.face_faiss_id IN ({placeholders})
        """, faiss_ids).fetchall()

