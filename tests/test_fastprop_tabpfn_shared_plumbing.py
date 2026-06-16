import json
import sys
import types

from apischema import deserialize


def _install_stub(name: str, attrs: dict):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _DummyEstimator:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _prepare_optional_runtime_stubs():
    _install_stub(
        "optunaz.algorithms.chem_prop",
        {
            "ChemPropRegressor": _DummyEstimator,
            "ChemPropClassifier": _DummyEstimator,
            "ChemPropRegressorPretrained": _DummyEstimator,
        },
    )
    _install_stub(
        "optunaz.algorithms.chem_prop_hyperopt",
        {
            "ChemPropHyperoptClassifier": _DummyEstimator,
            "ChemPropHyperoptRegressor": _DummyEstimator,
        },
    )
    _install_stub(
        "optunaz.algorithms.probabilistic_random_forest",
        {"PRFClassifier": _DummyEstimator},
    )
    _install_stub(
        "optunaz.algorithms.calibrated_cv",
        {"CalibratedClassifierCVWithVA": _DummyEstimator},
    )
    _install_stub(
        "optunaz.algorithms.mapie_uncertainty",
        {"MapieWithUncertainty": _DummyEstimator},
    )


_prepare_optional_runtime_stubs()

from optunaz.config.optconfig import OptimizationConfig
import optunaz.config.buildconfig as buildconfig
import optunaz.config.build_from_opt as build_from_opt
from optunaz.utils.files_paths import attach_root_path


def test_fastprop_tabpfn_shared_plumbing_examples():
    example_expectations = {
        "examples/optimization/FastProp_drd2_50.json": ["FastPropRegressor"],
        "examples/optimization/FastProp_drd2_50_cls.json": ["FastPropClassifier"],
        "examples/optimization/FastProp_drd2_50_covariate.json": ["FastPropRegressor"],
        "examples/optimization/FastProp_drd2_50_cls_covariate.json": ["FastPropClassifier"],
        "examples/optimization/TabPFN_drd2_50.json": ["TabPFNRegressor"],
        "examples/optimization/TabPFN_drd2_50_cls.json": ["TabPFNClassifier"],
    }

    for rel_path, expected in example_expectations.items():
        with open(attach_root_path(rel_path), "rt") as fp:
            cfg = deserialize(OptimizationConfig, json.load(fp))
        assert [type(alg).__name__ for alg in cfg.algorithms] == expected


def test_fastprop_tabpfn_shared_plumbing_hooks_present():
    assert hasattr(buildconfig, "FastPropRegressor")
    assert hasattr(buildconfig, "FastPropClassifier")
    assert hasattr(buildconfig, "TabPFNRegressor")
    assert hasattr(buildconfig, "TabPFNClassifier")
    assert hasattr(build_from_opt, "suggest_alg_params")
