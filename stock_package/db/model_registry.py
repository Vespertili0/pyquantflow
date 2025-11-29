from .base import BaseDB
import json
import datetime

class ModelRegistryDB(BaseDB):
    def _create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                version TEXT,
                created_at DATETIME,
                path TEXT,
                metrics TEXT
            )
        ''')
        self.conn.commit()

    def register_model(self, name, version, path, metrics=None):
        created_at = datetime.datetime.now().isoformat()
        metrics_json = json.dumps(metrics) if metrics else None
        self.cursor.execute('''
            INSERT INTO models (name, version, created_at, path, metrics)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, version, created_at, path, metrics_json))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_model(self, model_id):
        self.cursor.execute('SELECT * FROM models WHERE id = ?', (model_id,))
        return self.cursor.fetchone()
