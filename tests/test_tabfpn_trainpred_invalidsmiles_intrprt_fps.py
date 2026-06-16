import sys

import numpy as np
import pytest

if sys.platform == "win32":
    pytest.skip("TabPFN runtime verification is unavailable on Windows in this environment", allow_module_level=True)

try:
    import autogluon.tabular  # noqa: F401
    import tabpfn_extensions.post_hoc_ensembles.sklearn_interface  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"TabPFN runtime unavailable: {exc}", allow_module_level=True)

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

from optunaz.algorithms.tabpfn import TabPFNClassifier, TabPFNRegressor
from optunaz.datareader import Dataset
from optunaz.descriptors import (
    ECFP,
    UnscaledPhyschemDescriptors,
    descriptor_from_config,
)


@pytest.fixture
def file_drd2_50(shared_datadir):
    """Returns 50 molecules from DRD2 dataset."""
    return str(shared_datadir / "DRD2" / "subset-50" / "train.csv")


def test_tabpfn_trainpred(file_drd2_50):
    dataset = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=file_drd2_50,
    )
    X, y, _, _, _, _ = dataset.get_sets()
    descriptor, _ = descriptor_from_config(X, UnscaledPhyschemDescriptors.new())

    reg = TabPFNRegressor(max_time=30, max_feats=2)
    try:
        reg.fit(descriptor, y)
        preds = reg.predict(descriptor)
    except (ImportError, OSError) as exc:
        pytest.skip(f"TabPFN runtime unavailable: {exc}")
    assert np.all(np.isfinite(preds))



def test_tabpfn_trainpred_cls(file_drd2_50):
    dataset = Dataset(
        input_column="canonical",
        response_column="molwt_gt_330",
        training_dataset_file=file_drd2_50,
    )
    X, y, _, _, _, _ = dataset.get_sets()
    descriptor, _ = descriptor_from_config(X, ECFP.new())

    cls = TabPFNClassifier(max_time=30, max_feats=2)
    try:
        cls.fit(descriptor, y)
        preds = cls.predict(descriptor)
    except (ImportError, OSError) as exc:
        pytest.skip(f"TabPFN runtime unavailable: {exc}")
    assert np.all(np.isfinite(preds))



def test_tabpfn(file_drd2_50):
    dataset = Dataset(
        input_column="canonical",
        response_column="molwt",
        training_dataset_file=file_drd2_50,
    )
    X, y, _, _, _, _ = dataset.get_sets()
    descriptor, _ = descriptor_from_config(X, ECFP.new())

    try:
        reg = AutoTabPFNRegressor(max_time=30)
        reg.fit(descriptor[:, [875, 1152]], y)
        reg.predict(descriptor[:, [875, 1152]])
    except (ImportError, OSError) as exc:
        pytest.skip(f"TabPFN runtime unavailable: {exc}")
