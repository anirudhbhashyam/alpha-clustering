from pathlib import Path

from typing import NoReturn

import pytest

import numpy as np

from alpha_clustering.alpha_complex import AlphaComplex2D, AlphaComplex3D, AlphaComplexND
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
        ("test.arff", -0.4556, 12575),
        ("test_2.arff", -0.1054, 160),
        ("test_3.arff", -185.1124, 3770),
    ]
)
def test_alpha_complex_2d(data_file: Path, alpha: float, expected_simplices: int) -> NoReturn:
    test_data = TEST_IO.load_point_cloud(data_file)
    points = test_data.iloc[:, : -1].to_numpy()
    ac = AlphaComplex2D(points)
    ac.predict(alpha)
    assert ac.n_simplices == expected_simplices


@pytest.mark.parametrize(
    ("data_file", "alpha", "expected_simplices"),
    [
        ("golf-ball.arff", 0.8764, 47146),
        ("chainlink.arff", 5.00, 14162)
    ]
)
def test_alpha_complex_3d(data_file: Path, alpha: float, expected_simplices: int) -> NoReturn:
    test_data = TEST_IO.load_point_cloud(data_file)
    points = test_data.iloc[:, : -1].to_numpy()
    ac = AlphaComplex3D(points)
    ac.fit()
    ac.predict(alpha)
    assert ac.n_simplices == expected_simplices
    

@pytest.mark.parametrize(
    ("vertices", "alpha", "expected_simplices"),
    [
        (
            np.array(
                [[1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1]],
                dtype = np.float32
            ),
            1.00,
            11
        )
    ]
)
def test_alpha_complex_nd(vertices: np.ndarray, alpha: float, expected_simplices: int) -> NoReturn:
    ac = AlphaComplexND(vertices)
    ac.fit()
    ac.predict(alpha)
    assert ac.n_simplices == expected_simplices
    
