import itertools

from dataclasses import dataclass
from typing import NoReturn

from abc import ABC, abstractmethod

import alphashape

import numpy as np
import pandas as pd

from scipy.spatial import Delaunay, ConvexHull

from logger import Logger
from exceptions import AlphaValueError, NotFittedError

LOGGER = Logger(__name__)

@dataclass
class AlphaShape(ABC):
    vertices: np.ndarray
    alpha: float
    alpha_shape: tuple[np.ndarray] = None

    @abstractmethod
    def fit(self) -> None:
        pass

    @property
    def n_simplices(self) -> int:
        return sum(len(simplices) for simplices in self.alpha_shape)

    @property
    def get_shape(self) -> np.ndarray:
        if self.alpha_shape is None:
            raise NotFittedError(
                self.__class__.__name__,
                LOGGER
            )
        # if len(self.alpha_shape) == 0:
        #     raise AlphaValueError(
        #         self.alpha, 
        #         self.__class__.__name__,
        #         LOGGER
        #     )
        return self.alpha_shape
 
    @staticmethod
    @abstractmethod
    def _circum_radius(tessellation_vertices: np.ndarray) -> np.ndarray:
        pass

    def _face_filter(self, simplices: np.ndarray) -> np.ndarray:
        k = simplices.shape[1]
        simplex_filter = list(itertools.combinations(list(range(k)), k - 1))
        faces = simplices[:, simplex_filter].reshape(-1, k - 1)
        faces = np.sort(faces, axis = 1)
        return np.unique(faces, axis = 0)

class AlphaShape2D(AlphaShape):
    @staticmethod
    def _circum_radius(tessellation_vertices: np.ndarray) -> np.ndarray:
        def _circum_radius_helper(tessellation_vertices: np.ndarray) -> np.ndarray:
            _m = np.concatenate(
                [
                    tessellation_vertices, 
                    np.ones(tessellation_vertices.shape[: 2] + (1, ))
                ],
                axis = -1
            )
            return 0.5 * np.linalg.det(_m) 

        side_a_arr = np.linalg.norm(tessellation_vertices[:, 0, :] - tessellation_vertices[:, 1, :], axis = -1)
        side_b_arr = np.linalg.norm(tessellation_vertices[:, 1, :] - tessellation_vertices[:, 2, :], axis = -1)
        side_c_arr = np.linalg.norm(tessellation_vertices[:, 0, :] - tessellation_vertices[:, 2, :], axis = -1)
        triangle_areas = _circum_radius_helper(tessellation_vertices)

        return (side_a_arr * side_b_arr * side_c_arr) / (4 * triangle_areas)

    def fit(self) -> None:
        LOGGER.info(
            f"Finding the \u03B1-shape for given point set with \u03B1 = {self.alpha}..."
        )
        if self.alpha == 0:
            self.alpha_shape = ConvexHull(self.vertices).simplices
            self.alpha_shape = np.concatenate(
                [
                    self.alpha_shape, 
                    np.zeros(self.alpha_shape.shape[0], dtype = np.int32)[..., None]
                ],
                axis = -1
            )
            return        

        one_by_alpha  = 1 / self.alpha
        furthest_site = True if self.alpha > 0 else False
        tessellation  = Delaunay(self.vertices, furthest_site = furthest_site)
        LOGGER.info(
            f"Constructed the delaunay triangulation with {furthest_site = } and {tessellation.nsimplex} simplices."
        )
        simplices = self.vertices.take(tessellation.simplices, axis = 0)
        simplices_circum_radii = self._circum_radius(simplices)
        bool_index = (simplices_circum_radii >= one_by_alpha) if furthest_site else (simplices_circum_radii <= -one_by_alpha)
        picked_simplex_indices = np.concatenate([np.where(bool_index)[0]])
        picked_simplices = tessellation.simplices[picked_simplex_indices]
        unique_triangles = np.unique(picked_simplices, axis = 0)
        unique_edges = self._face_filter(unique_triangles)

        self.alpha_shape = (unique_triangles, unique_edges)
        
        LOGGER.info(
            f"\u03B1-shape with {self.n_simplices} simplices generated."
        )

    def get_summary(self, dataset: str) -> pd.DataFrame:
        summary = {
            "Alpha": self.alpha,
            "Number of vertices": len(self.vertices),
            "Number of simplices": len(self.alpha_shape),
        }

        return pd.DataFrame.from_dict(
            summary,
            orient = "index",
            columns = [dataset]
        )
        
@dataclass
class AlphaShape3D(AlphaShape):
    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise AlphaValueError(
                self.alpha, 
                self.__class__.__name__,
                LOGGER
            )

    @staticmethod
    def _circum_radius(tessellation_vertices: np.ndarray) -> np.ndarray:
        def _circum_radius_helper(tessellation_vertices: np.ndarray) -> \
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray
        ]:
            vertex_norms = (np.linalg.norm(tessellation_vertices, axis = -1) ** 2)[:, :, None]
            _m = np.concatenate(
                [
                    tessellation_vertices, 
                    vertex_norms,
                    np.ones(tessellation_vertices.shape[: -1] + (1, ))
                ],
                axis = -1
            )
            _m1234 = _m[:, :, [1, 2, 3, 4]]
            _m0234 = _m[:, :, [0, 2, 3, 4]]
            _m0134 = _m[:, :, [0, 1, 3, 4]]
            _m0124 = _m[:, :, [0, 1, 2, 4]]
            _m0123 = _m[:, :, [0, 1, 2, 3]]
            return (
                np.linalg.det(_m1234),
                np.linalg.det(_m0234),
                np.linalg.det(_m0134),
                np.linalg.det(_m0124),
                np.linalg.det(_m0123)
            )

        _m1234_dets, _m0234_dets, \
        _m0134_dets, _m0124_dets, _m0123_dets = _circum_radius_helper(tessellation_vertices)
        num = _m1234_dets ** 2 + _m0234_dets ** 2 + _m0134_dets ** 2 + 4 * _m0124_dets * _m0123_dets
        dem = 2 * abs(_m0124_dets)
        return np.sqrt(num) / dem

    def fit(self) -> None:
        if self.alpha == 0:
            self.alpha_shape = ConvexHull(self.vertices).simplices
            return
        tessellation = Delaunay(self.vertices, furthest_site = False)
        LOGGER.info(
            f"Constructed the delaunay triangulation with furthest_site = False and {tessellation.nsimplex} simplices."
        )
        simplices = self.vertices.take(tessellation.simplices, axis = 0)
        simplices_circum_radii = self._circum_radius(simplices)
        bool_index = (simplices_circum_radii <= (1 / self.alpha))
        picked_simplex_indices = np.concatenate([np.where(bool_index)[0]])
        picked_simplices = tessellation.simplices[picked_simplex_indices]

        unique_triangles = self._face_filter(picked_simplices)

        unique_edges = self._face_filter(unique_triangles)

        self.alpha_shape = (unique_triangles, unique_edges)

        LOGGER.info(
            f"\u03B1-shape with {self.n_simplices} simplices generated."
        )

        # Vertices = np.unique(Edges)
        # return Vertices, Edges, Triangles

    def get_summary(self, dataset: str) -> pd.DataFrame:
        summary = {
            "Alpha": self.alpha,
            "Number of vertices": len(self.vertices),
            "Number of simplices": self.n_simplices,
        }

        return pd.DataFrame.from_dict(
            summary,
            orient = "index",
            columns = [dataset]
        )

class AlphaShapeND(AlphaShape):
    def fit(self) -> None:
        self.alpha_shape = alphashape.alphashape(self.vertices, self.alpha)

    def get_summary(self, dataset: str) -> pd.DataFrame:
        summary = {
            "Alpha": self.alpha,
            "Number of vertices": len(self.vertices),
            "Number of simplices": len(self.alpha_shape),
        }

        return pd.DataFrame.from_dict(
            summary,
            orient = "index",
            columns = [dataset]
        )

    @staticmethod
    def _circum_radius(tessellation_vertices: np.ndarray) -> NoReturn:
        raise NotImplementedError("Not implemented for ND alpha shapes.")