"""RandomForest wrapper for video summarization scoring."""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class VideoRandomForest:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "VideoRandomForest":
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        return instance
