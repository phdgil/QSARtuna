import glob
import importlib
import importlib.util
import io
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted


def _load_tabpfn_runtime():
    try:
        tabular = importlib.import_module("autogluon.tabular")
        sklearn_interface = importlib.import_module(
            "tabpfn_extensions.post_hoc_ensembles.sklearn_interface"
        )
    except (ImportError, OSError) as exc:
        raise ImportError(
            "TabPFN runtime requires autogluon.tabular and "
            "tabpfn_extensions.post_hoc_ensembles.sklearn_interface"
        ) from exc

    return (
        tabular.TabularPredictor,
        sklearn_interface.AutoTabPFNClassifier,
        sklearn_interface.AutoTabPFNRegressor,
    )



def save_model_memory(model_dir):
    """Saves the model directory as a tarball in memory."""
    tarblob = io.BytesIO()
    with tarfile.TarFile(mode="w", fileobj=tarblob) as tar:
        dirinfo = tarfile.TarInfo(model_dir)
        dirinfo.mode = 0o755
        dirinfo.type = tarfile.DIRTYPE
        tar.addfile(dirinfo, None)
        for dirpath, _, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(dirpath, file)
                with open(file_path, "rb") as fh:
                    filedata = io.BytesIO(fh.read())
                    fileinfo = tarfile.TarInfo(str(file_path))
                    fileinfo.size = len(filedata.getbuffer())
                    tar.addfile(fileinfo, filedata)
    return tarblob


def extract_model_memory(tarblob, temp_dir, save_dir):
    """Extracts the model directory from a tarball in memory."""
    tarblob.seek(0)
    with tarfile.TarFile(mode="r", fileobj=tarblob) as tar:
        for member in tar.getmembers():
            member.name = os.path.relpath(member.name, save_dir)
            tar.extract(member, temp_dir)
    return


@contextmanager
def suppress_logging(level=logging.FATAL):
    """Silence TabPFN outputs."""
    logger = logging.getLogger()
    previous_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


def _default_tabpfn_cache_dir() -> str:
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Caches/tabpfn/")
    if sys.platform == "win32":
        return os.path.join(
            os.getenv("LOCALAPPDATA", tempfile.gettempdir()), "QSARtuna", "tabpfn"
        )
    return os.path.expanduser("~/.cache/tabpfn/")


def prep_alg(env_path: str, model_type: str) -> None:
    """Prepare TabPFN if necessary by copying checkpoint files into the extension cache."""

    source_dir = os.getenv(env_path, _default_tabpfn_cache_dir())
    source_files = glob.glob(os.path.join(source_dir, f"tabpfn-v2-{model_type}*.ckpt"))

    spec = importlib.util.find_spec("tabpfn_extensions")
    if not spec or not spec.submodule_search_locations:
        raise ImportError("tabpfn_extensions package not found")
    dest_dir = spec.submodule_search_locations[0]

    for file in source_files:
        dest_path = os.path.join(dest_dir, "hpo", "hpo_models", os.path.basename(file))
        if not os.path.exists(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(file, dest_path)
    return


def set_device(mps_fallback: str = "cpu") -> str:
    """Select a runtime device across supported platforms."""
    try:
        from torch import cuda
        from torch.backends import mps
    except ImportError:
        return "cpu"

    return mps_fallback if mps.is_available() else "cuda" if cuda.is_available() else "cpu"


class BaseTabPFN(BaseEstimator):
    """Base class for TabPFN models with shared runtime management."""

    def __init__(
        self,
        max_time: int = 30,
        random_state: int | np.random.RandomState = 42,
        max_feats: int = 500,
        feature_selection: Literal["k_best", "tree"] = "k_best",
        eval_metric: str | None = None,
    ):
        self.max_time = max_time
        self.random_state = random_state
        self.max_feats = max_feats
        self.feature_selection = feature_selection
        self.eval_metric = eval_metric

    def _transform_features(self, X):
        X = check_array(X, ensure_2d=True, allow_nd=False)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]} features."
            )
        if hasattr(self, "_important_feats"):
            X = X[:, self._important_feats]
        return X

    def _load_predictor(self, tmpdir):
        tabular_predictor, _, _ = _load_tabpfn_runtime()
        extract_model_memory(self.model_tar_, tmpdir, self.output_dir_)
        self.model_.predictor_ = tabular_predictor.load(tmpdir)
        return self.model_.predictor_

    @staticmethod
    def _to_numpy(preds):
        if isinstance(preds, (pd.Series, pd.DataFrame)):
            return preds.to_numpy()
        return np.asarray(preds)

    def fit(self, X, y):
        X = check_array(X, ensure_2d=True, allow_nd=False)
        y = check_array(y, ensure_2d=False, allow_nd=False)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]

        _, classifier_class, regressor_class = _load_tabpfn_runtime()
        model_class, prep_path, feat_alg, score_func = (
            (
                classifier_class,
                "TABPFN_CLS__PATH",
                RandomForestClassifier,
                f_classif,
            )
            if is_classifier(self)
            else (
                regressor_class,
                "TABPFN_REG__PATH",
                RandomForestRegressor,
                f_regression,
            )
        )
        ignore_limits = len(y) >= 10000
        phe_init_args = {"SUBSAMPLE_SAMPLES": 10000} if ignore_limits else {}

        with tempfile.TemporaryDirectory() as output_dir:
            phe_init_args["path"] = output_dir
            self.output_dir_ = output_dir
            self.model_ = model_class(
                max_time=self.max_time,
                random_state=self.random_state,
                eval_metric=self.eval_metric,
                ignore_pretraining_limits=ignore_limits,
                phe_init_args=phe_init_args,
            )

            prep_alg(prep_path, "classifier" if is_classifier(self) else "regressor")

            if is_classifier(self):
                self.classes_ = unique_labels(y)

            if X.shape[1] > self.max_feats:
                self._selector = (
                    SelectKBest(score_func=score_func, k=self.max_feats).fit(X, y)
                    if self.feature_selection == "k_best"
                    else feat_alg(random_state=self.random_state).fit(X, y)
                )
                self._important_feats = (
                    self._selector.get_support(indices=True)
                    if self.feature_selection == "k_best"
                    else self._selector.feature_importances_.argsort()[-self.max_feats :]
                )
                X = X[:, self._important_feats]

            self.model_.device = set_device()
            self.model_.ignore_pretraining_limits = (
                self.model_.ignore_pretraining_limits or self.max_feats > 500
            )
            with suppress_logging():
                self.model_.fit(X, y)
                self.model_tar_ = save_model_memory(output_dir)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["model_", "model_tar_"])
        X = self._transform_features(X)
        self.model_.device = set_device()
        with suppress_logging():
            with tempfile.TemporaryDirectory() as tmpdir:
                self._load_predictor(tmpdir)
                preds = (
                    self.model_.predict_proba(X)
                    if is_classifier(self)
                    else self.model_.predict(X)
                )
        return self._to_numpy(preds)

    def predict(self, X):
        if is_classifier(self):
            probabilities = self.predict_proba(X)
            if probabilities.ndim == 1:
                return probabilities > 0.5
            return self.classes_[np.argmax(probabilities, axis=1)]

        predictions = self.predict_proba(X).flatten()
        return (
            predictions.clip(0, 1)
            if 0 <= self.y_.min() <= 1 and 0 <= self.y_.max() <= 1
            else predictions
        )

    def _predict_uncert(self, X):
        """Predict uncertainty for regression or classification tasks."""
        check_is_fitted(self, ["model_", "model_tar_"])
        X = self._transform_features(X)
        self.model_.device = set_device()
        with suppress_logging():
            with tempfile.TemporaryDirectory() as tmpdir:
                predictor = self._load_predictor(tmpdir)
                model_names = predictor.model_names()
                cols = [f"f{i}" for i in range(X.shape[1])]
                X_df = pd.DataFrame(X, columns=cols)
                preds = []
                for model_name in model_names:
                    if is_classifier(self):
                        preds.append(self._to_numpy(predictor.predict_proba(X_df, model=model_name)))
                    else:
                        preds.append(self._to_numpy(predictor.predict(X_df, model=model_name)))
        preds = np.asarray(preds)
        if is_classifier(self) and preds.ndim == 3:
            return preds.std(axis=0).mean(axis=1)
        return preds.std(axis=0)


class TabPFNRegressor(RegressorMixin, BaseTabPFN):
    """Sklearn-like TabPFN regressor."""

    def __init__(
        self,
        max_time: int = 30,
        random_state: int | np.random.RandomState = 42,
        max_feats: int = 500,
        feature_selection: Literal["k_best", "tree"] = "k_best",
        eval_metric: Literal[
            "root_mean_squared_error", "mse", "mae"
        ] = "root_mean_squared_error",
    ):
        self.max_time = max_time
        self.random_state = random_state
        self.max_feats = max_feats
        self.feature_selection = feature_selection
        self.eval_metric = eval_metric


class TabPFNClassifier(ClassifierMixin, BaseTabPFN):
    """Sklearn-like TabPFN classifier."""

    def __init__(
        self,
        max_time: int = 30,
        random_state: int | np.random.RandomState = 42,
        max_feats: int = 500,
        feature_selection: Literal["k_best", "tree"] = "k_best",
        eval_metric: Literal["accuracy", "roc_auc", "f1", "log_loss"] = "accuracy",
    ):
        self.max_time = max_time
        self.random_state = random_state
        self.max_feats = max_feats
        self.feature_selection = feature_selection
        self.eval_metric = eval_metric
