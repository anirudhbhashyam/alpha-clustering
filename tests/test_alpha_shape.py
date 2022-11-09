from pathlib import Path

from typing import NoReturn

import pytest

from alpha_clustering.alpha_shape import AlphaShape2D, AlphaShape3D
from alpha_clustering.config import Config
from alpha_clustering.io_handler import IOHandler

CPD = Path(__file__).resolve().parents[1]

TEST_CONFIG = Config(
    config_dir = CPD / "config",
    filenames = ["settings.json"]
)

TEST_CONFIG_DATA = next(TEST_CONFIG.load())

TEST_IO = IOHandler(
    data_dir = Path(TEST_CONFIG_DATA["PATH"]["DATA_POINTS_PATH"]).resolve(),
    save_dir = CPD / "tests" / "results"
)


@pytest.mark.parametrize(
    ("data_file", "alpha", "expected_simplices"),
    [
        ("test.arff", -0.4556, 14320),
        ("test_2.arff", -0.1054, 10966),
        ("test_3.arff", -185.1124, 2898),
    ]
)
def test_alpha_shape_2d(data_file: Path, alpha: float, expected_simplices: int) -> NoReturn:
    test_data = TEST_IO.load_point_cloud(data_file)
    points = test_data.iloc[:, : -1].to_numpy()
    ac = AlphaShape2D(points, alpha)
    ac.fit()
    assert ac.n_simplices == expected_simplices


@pytest.mark.parametrize(
    ("data_file", "alpha", "expected_simplices"),
    [
        ("golf-ball.arff", 0.8764, 47146),
        ("chainlink.arff", 5.00, 14162)
    ]
)
def test_alpha_shape_3d(data_file: Path, alpha: float, expected_simplices: int) -> NoReturn:
    test_data = TEST_IO.load_point_cloud(data_file)
    points = test_data.iloc[:, : -1].to_numpy()
    ac = AlphaShape3D(points, alpha)
    ac.fit()
    assert ac.n_simplices == expected_simplices
