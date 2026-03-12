from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def zc_count(x: np.ndarray, thr: float = 1e-4) -> float:
    x1 = x[:-1]
    x2 = x[1:]
    return float(np.sum(((x1 * x2) < 0) & (np.abs(x1 - x2) >= thr)))


def ssc_count(x: np.ndarray, thr: float = 1e-4) -> float:
    x_prev = x[:-2]
    x_mid = x[1:-1]
    x_next = x[2:]
    return float(np.sum((((x_mid - x_prev) * (x_mid - x_next)) > 0) &
                        ((np.abs(x_mid - x_prev) >= thr) | (np.abs(x_mid - x_next) >= thr))))


def extract_window_features(window_2d: np.ndarray) -> np.ndarray:
    # window_2d: [window_size, n_channels]
    feats = []
    for ch in range(window_2d.shape[1]):
        s = window_2d[:, ch]
        mav = np.mean(np.abs(s))
        rms = np.sqrt(np.mean(s ** 2))
        wl = np.sum(np.abs(np.diff(s)))
        zc = zc_count(s)
        ssc = ssc_count(s)
        var = np.var(s)
        iemg = np.sum(np.abs(s))
        mean = np.mean(s)
        std = np.std(s)
        min_v = np.min(s)
        max_v = np.max(s)
        ptp = max_v - min_v
        # Use quantile with linear interpolation instead of percentile to avoid hanging
        s_sorted = np.sort(s)
        n = len(s_sorted)
        p25_idx = int(np.ceil(0.25 * n)) - 1
        p50_idx = int(np.ceil(0.50 * n)) - 1
        p75_idx = int(np.ceil(0.75 * n)) - 1
        p25 = s_sorted[max(0, p25_idx)]
        p50 = s_sorted[max(0, p50_idx)]
        p75 = s_sorted[max(0, p75_idx)]
        feats.extend([
            mav,
            rms,
            wl,
            zc,
            ssc,
            var,
            iemg,
            mean,
            std,
            min_v,
            max_v,
            ptp,
            p25,
            p50,
            p75,
        ])
    return np.asarray(feats, dtype=np.float32)


@dataclass
class EMGLinearModel:
    scaler: StandardScaler | None
    clf: object

    @staticmethod
    def create(
        model_type: str = "extra_trees",
        C: float = 1.0,
        max_iter: int = 5000,
        n_estimators: int = 600,
        max_depth: int | None = None,
        random_state: int = 42,
    ):
        if model_type == "logreg":
            scaler = StandardScaler()
            common = dict(C=C, max_iter=max_iter, solver="lbfgs")
            try:
                clf = LogisticRegression(multi_class="multinomial", **common)
            except TypeError:
                clf = LogisticRegression(**common)
            return EMGLinearModel(scaler=scaler, clf=clf)

        if model_type == "extra_trees":
            clf = ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
            return EMGLinearModel(scaler=None, clf=clf)

        raise ValueError(f"Unsupported model_type: {model_type}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        if self.scaler is None:
            self.clf.fit(X_train, y_train)
            return
        Xs = self.scaler.fit_transform(X_train)
        self.clf.fit(Xs, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return self.clf.predict(X)
        Xs = self.scaler.transform(X)
        return self.clf.predict(Xs)