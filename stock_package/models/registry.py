from ..db.model_registry import ModelRegistryDB
from ..config import MODEL_REGISTRY_DB
import os
import shutil

class ModelRegistry:
    def __init__(self, db_path=MODEL_REGISTRY_DB, storage_dir=None):
        self.db_path = db_path
        self.storage_dir = storage_dir or os.path.join(os.path.dirname(db_path), 'model_files')
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def register(self, name, version, model_path, metrics=None):
        # Copy model file to storage dir with a unique name or timestamp
        filename = os.path.basename(model_path)
        dest_path = os.path.join(self.storage_dir, f"{name}_{version}_{filename}")
        shutil.copy2(model_path, dest_path)

        with ModelRegistryDB(self.db_path) as db:
            model_id = db.register_model(name, version, dest_path, metrics)
            print(f"Model registered with ID: {model_id}")
            return model_id

    def get_model_details(self, model_id):
        with ModelRegistryDB(self.db_path) as db:
            return db.get_model(model_id)
