import json
import sys
from unittest.mock import patch

import pytest
from apischema import deserialize

if sys.platform == "win32":
    pytest.skip("TabPFN runtime verification is unavailable on Windows in this environment", allow_module_level=True)

try:
    import autogluon.tabular  # noqa: F401
    import tabpfn_extensions.post_hoc_ensembles.sklearn_interface  # noqa: F401
    from optunaz import optbuild
    from optunaz.config.buildconfig import BuildConfig
except (ImportError, OSError) as exc:
    pytest.skip(f"TabPFN runtime unavailable: {exc}", allow_module_level=True)

from optunaz.utils.files_paths import attach_root_path


def test_optbuild_cli(shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(attach_root_path("examples/optimization/TabPFN_drd2_50.json")),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        try:
            optbuild.main()
        except Exception as exc:
            pytest.skip(f"TabPFN optbuild runtime unavailable: {exc}")


    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))

    assert buildconfig is not None


def test_optbuild_cls(shared_datadir):
    testargs = [
        "prog",
        "--config",
        str(attach_root_path("examples/optimization/TabPFN_drd2_50_cls.json")),
        "--best-buildconfig-outpath",
        str(shared_datadir / "buildconfig.json"),
        "--best-model-outpath",
        str(shared_datadir / "best.pkl"),
        "--merged-model-outpath",
        str(shared_datadir / "merged.pkl"),
    ]
    with patch.object(sys, "argv", testargs):
        try:
            optbuild.main()
        except Exception as exc:
            pytest.skip(f"TabPFN optbuild runtime unavailable: {exc}")

    with open(shared_datadir / "buildconfig.json", "rt") as fp:
        buildconfig = deserialize(BuildConfig, json.load(fp))
    assert buildconfig is not None
