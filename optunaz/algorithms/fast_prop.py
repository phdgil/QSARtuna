import io
import logging
import os
import re
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


logger = logging.getLogger(__name__)


@contextmanager
def suppress_logging(level=logging.CRITICAL):
    previous_level = logging.getLogger().level
    logging.getLogger().setLevel(level)
    try:
        yield
    finally:
        logging.getLogger().setLevel(previous_level)


def _require_fastprop():
    try:
        from fastprop.cli.predict import predict_fastprop
        from fastprop.cli.train import train_fastprop
    except (ImportError, OSError) as exc:
        raise ImportError(
            "FastProp runtime requires the optional fastprop/torch stack to import cleanly"
        ) from exc
    return train_fastprop, predict_fastprop


def save_model_memory(model_dir):
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
    tarblob.seek(0)
    with tarfile.TarFile(mode="r", fileobj=tarblob) as tar:
        for member in tar.getmembers():
            member.name = os.path.relpath(member.name, save_dir)
            tar.extract(member, temp_dir)


class BaseFastProp(BaseEstimator):
    def __init__(
        self,
        fnn_layers: int = 2,
        learning_rate: float = 0.0001,
        batch_size: int = 2048,
        number_epochs: int = 30,
        number_repeats: int = 1,
        train_size: float = 0.8,
        val_size: float = 0.15,
        test_size: float = 0.05,
        random_seed: int = 42,
        hidden_size: int = 1800,
        patience: int = 5,
    ):
        self.fnn_layers = fnn_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.number_epochs = number_epochs
        self.number_repeats = number_repeats
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_seed = random_seed
        self.hidden_size = hidden_size
        self.patience = patience

    def fit(self, X, y, sample_weight=None):
        train_fastprop, _ = _require_fastprop()
        X = np.asarray(X)
        y = np.asarray(y)

        self.X_ = X
        self.y_ = y
        self.sample_weight_ = sample_weight

        if is_classifier(self):
            self.classes_ = unique_labels(y).astype(np.uint8)
            y = y.astype(np.uint8)

        descriptor_columns = list(map(str, range(self.X_.shape[1])))

        with tempfile.TemporaryDirectory() as output_dir:
            self.output_dir_ = output_dir

            pd.DataFrame(self.y_, columns=["targets"]).to_csv(
                os.path.join(output_dir, "input_file.csv"), index_label="Index"
            )
            pd.DataFrame(self.X_, columns=descriptor_columns).to_csv(
                os.path.join(output_dir, "precomputed.csv"), index_label="Index"
            )

            with suppress_logging():
                train_fastprop(
                    output_directory=Path(output_dir),
                    input_file=Path(os.path.join(output_dir, "input_file.csv")),
                    smiles_column="Index",
                    descriptor_set="all",
                    target_columns=["targets"],
                    enable_cache=False,
                    precomputed=Path(os.path.join(output_dir, "precomputed.csv")),
                    fnn_layers=self.fnn_layers,
                    hidden_size=self.hidden_size,
                    clamp_input=False,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    number_epochs=self.number_epochs,
                    number_repeats=self.number_repeats,
                    problem_type="binary" if is_classifier(self) else "regression",
                    train_size=self.train_size,
                    val_size=self.val_size,
                    test_size=self.test_size,
                    sampler="random",
                    random_seed=self.random_seed,
                    patience=self.patience,
                    standardize=False,
                    hopt=False,
                )

            self.model_ = save_model_memory(output_dir)
            self.checkpoint_dir_ = next(
                directory
                for directory in os.listdir(output_dir)
                if re.match(r"^fastprop_.*", directory)
            )
        return self

    def predict_proba(self, X):
        _, predict_fastprop = _require_fastprop()
        check_is_fitted(self, ["model_"])
        X = np.asarray(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            precomputed_path = os.path.join(tmpdir, "precomputed_descriptors.csv")
            preds_path = os.path.join(tmpdir, "preds.csv")

            extract_model_memory(self.model_, tmpdir, self.output_dir_)
            pd.DataFrame(X, columns=list(map(str, range(X.shape[1])))).to_csv(
                precomputed_path, index_label="Index"
            )

            predict_fastprop(
                checkpoints_dir=Path(f"{tmpdir}/{self.checkpoint_dir_}/checkpoints"),
                smiles_strings=range(len(X)),
                descriptor_set="all",
                standardize=False,
                smiles_file=None,
                precomputed_descriptors=Path(precomputed_path),
                output=preds_path,
            )

            preds = pd.read_csv(preds_path)[["task_0"]].values

        if is_classifier(self):
            probabilities = np.zeros([len(preds), 2])
            probabilities[:, 1] = preds.flatten()
            probabilities[:, 0] = 1 - probabilities[:, 1]
            return probabilities

        return preds.flatten()

    def predict(self, X):
        if is_classifier(self):
            return self.predict_proba(X)[:, 1] > 0.5

        predictions = self.predict_proba(X).flatten()
        if 0 <= self.y_.min() <= 1 and 0 <= self.y_.max() <= 1:
            return predictions.clip(0, 1)
        return predictions

    def predict_uncert(self, X):
        _, predict_fastprop = _require_fastprop()
        check_is_fitted(self, ["model_"])
        X = np.asarray(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            precomputed_path = os.path.join(tmpdir, "precomputed_descriptors.csv")
            preds_path = os.path.join(tmpdir, "preds.csv")

            extract_model_memory(self.model_, tmpdir, self.output_dir_)
            pd.DataFrame(X, columns=list(map(str, range(X.shape[1])))).to_csv(
                precomputed_path, index_label="Index"
            )

            predict_fastprop(
                checkpoints_dir=Path(f"{tmpdir}/{self.checkpoint_dir_}/checkpoints"),
                smiles_strings=range(len(X)),
                descriptor_set="all",
                standardize=False,
                smiles_file=None,
                precomputed_descriptors=Path(precomputed_path),
                output=preds_path,
            )

            preds = pd.read_csv(preds_path)[["task_0", "task_0_stdev"]].values

        return preds[:, 0], preds[:, 1]


class FastPropRegressor(RegressorMixin, BaseFastProp):
    def __str__(self):
        do_not_print = {"X_", "y_", "model_"}
        attributes = [
            f"{key}='{value}'"
            for key, value in self.__dict__.items()
            if key not in do_not_print
        ]
        return f"FastPropRegressor({', '.join(attributes)})"


class FastPropClassifier(ClassifierMixin, BaseFastProp):
    def __str__(self):
        do_not_print = {"X_", "y_", "model_"}
        attributes = [
            f"{key}='{value}'"
            for key, value in self.__dict__.items()
            if key not in do_not_print
        ]
        return f"FastPropClassifier({', '.join(attributes)})"
