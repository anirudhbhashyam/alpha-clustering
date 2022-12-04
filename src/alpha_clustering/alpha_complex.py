import itertools

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from scipy.spatial import Delaunay, ConvexHull

from .logger import Logger
from .exceptions import AlphaValueError, NotFittedError

LOGGER = Logger(__name__)

class AlphaComplex(ABC):
    """
    An abstract base class for Alpha Complexes. Works like sklearn classes with a fit and predict method.
    """
    vertices: np.ndarray
    tesselation: np.ndarray
    alpha_complex: list[np.array]

    @abstractmethod
    def __init__(self, vertices: np.ndarray) -> None:
        """
        Initialises an Alpha Complex interface. The implementing interface should only receive the vertices the complex needs to be constructed during initialisation.
        """
        pass

    @abstractmethod
    def fit(self) -> None:
        """
        The fit method of a class implementing this base class, usually constructs and stores the Delaunay triangulation.
        """
        pass

    @abstractmethod
    def predict(self, alpha: float) -> list[np.array]:
        """
        The predict method of a class that implements this base class, constructs the complex by filtering the Delaunay triangulation. 
        """
        pass

    @property
    def n_simplices(self) -> int:
        """
        A convenience property that calculates the total number of simplices generated in an Alpha complex.
        """
        return int(sum(len(simplices) for simplices in self.alpha_complex))

    @property
    def get_complex(self) -> np.ndarray:
        """
        A convenience property that returns the generated Alpha complex checking if it has been constructed.
        """
        if self.alpha_complex is None:
            raise NotFittedError(
                type(self),
                LOGGER
            )
        return self.alpha_complex
 
    @staticmethod
    @abstractmethod
    def _circum_radius(tessellation_vertices: np.ndarray) -> np.ndarray:
        """
        A private method that is implemented by a derived class that provides functionality to calculate the circumradii of simplices.
        """
        pass

    def _face_filter(self, simplices: np.ndarray) -> np.ndarray:
        """
        A private method that is provided to the derived class to filter down simplices in dimension and provide unique sets of the same.
        """
        k = simplices.shape[1]
        simplex_filter = list(itertools.combinations(list(range(k)), k - 1))
        faces = simplices[:, simplex_filter].reshape(-1, k - 1)
        faces = np.sort(faces, axis = 1)
        return np.unique(faces, axis = 0)


class AlphaComplex2D(AlphaComplex):
    """
    A specialised class that implements `AlphaComplex` that constructs 2D Alpha complexes.
    """
    def __init__(self, vertices: np.ndarray) -> None:
        self.vertices = vertices
        self.tesselation = None
        self.alpha_complex = None

    def fit(self) -> None:
        """
        `AlphaComplex2D` does not implement the fit method as the Delaunay triangulation needs to be reconstructed based on the value of `alpha`.

        Raises
        ------
        NotImplementedError:
            If the method is called.
        """
        raise NotImplementedError(
            "AlphaComplex2D does not support fit method. Please call predict directly."
        )

    def predict(self, alpha: float) -> None:
        """
        Creates the Alpha complex by filtering the Delaunay triangulation.

        Parameters
        ----------
        alpha: 
            Parameterises the constructed complex.
        """
        LOGGER.info(
            f"Finding the \u03B1-complex for given point set with \u03B1 = {alpha}..."
        )
        if alpha == 0:
            simplices = ConvexHull(self.vertices).simplices
            self.alpha_complex = (
                    np.concatenate(
                    [
                        simplices, 
                        np.zeros(simplices.complex[0], dtype = np.int32)[..., None]
                    ],
                    axis = -1
                ),
                simplices
            )
            return   
        furthest_site = True if alpha > 0 else False
        self.tesselation = Delaunay(self.vertices, furthest_site = furthest_site)
        LOGGER.info(
            f"Constructed the delaunay triangulation with furthest_site = False and {len(self.tesselation.simplices)} simplices."
        )     
        one_by_alpha  = 1 / alpha
        simplices = self.vertices.take(self.tesselation.simplices, axis = 0)
        simplices_circum_radii = self._circum_radius(simplices)
        bool_index = (simplices_circum_radii >= one_by_alpha) if furthest_site else (simplices_circum_radii <= -one_by_alpha)
        picked_simplex_indices = np.concatenate([np.where(bool_index)[0]])
        picked_simplices = self.tesselation.simplices[picked_simplex_indices]
        unique_triangles = np.unique(picked_simplices, axis = 0)
        unique_edges = self._face_filter(unique_triangles)

        self.alpha_complex = [unique_triangles, unique_edges]
        
        LOGGER.info(
            f"\u03B1-complex with {self.n_simplices} simplices generated."
        )

    @staticmethod
    def _circum_radius(tessellation_vertices: np.ndarray) -> np.array:
        """
        Calculates the circum radii of simplices using the areas of triangles.

        Parameters
        ----------
        tesselation_vertices:
            A numpy ndarray containing the simplices. Usually of the complex (`n, k + 1, k`). This usually denotes that there are `n` simplices each with dimension `k` and number of points `k + 1`.

        Returns
        -------
        ``numpy array``:
            Array containing the circum radii of the `n` simplices.
        """
        def _circum_radius_helper(tessellation_vertices: np.ndarray) -> np.ndarray:
            _m = np.concatenate(
                [
                    tessellation_vertices, 
                    np.ones(tessellation_vertices.shape[: 2] + (1, ))
                ],
                axis = -1
            )
            return 0.5 * np.linalg.det(_m) 

        side_a_arr = np.linalg.norm(tessellation_vertices[:, 0, :] - tessellation_vertices[:, 1, :], axis = -1) ** 2
        side_b_arr = np.linalg.norm(tessellation_vertices[:, 1, :] - tessellation_vertices[:, 2, :], axis = -1) ** 2
        side_c_arr = np.linalg.norm(tessellation_vertices[:, 0, :] - tessellation_vertices[:, 2, :], axis = -1) ** 2
        triangle_areas = _circum_radius_helper(tessellation_vertices)

        return (side_a_arr * side_b_arr * side_c_arr) / (4 * triangle_areas ** 2)

    def get_summary(self, alpha: float, dataset: str) -> pd.DataFrame:
        """
        Produces a summary of the generated complex.

        Parameters
        ----------
        alpha:
            The value used to construct the Alpha complex.

        dataset:
            Name of the dataset on which the complex was constructed.

        Returns
        -------
        ``pandas DataFrame``:
            A dataframe of values.
        """
        summary = {
            "Alpha": alpha,
            "Number of vertices": int(len(self.vertices)),
            "Number of simplices": self.n_simplices,
        }

        return pd.DataFrame.from_dict(
            summary,
            orient = "index",
            columns = [f"Dataset: {dataset}"]
        )
        

class AlphaComplex3D(AlphaComplex):
    """
    A specialised class that implements `AlphaComplex` that constructs 3D Alpha complexes.
    """
    def __init__(self, vertices: np.ndarray) -> None:
        self.vertices = vertices
        self.tesselation = None
        self.alpha_complex = None

    def fit(self) -> None:
        """
        Constructs the Delaunay triangulation of the point cloud.
        """
        self.tesselation = Delaunay(self.vertices)
        LOGGER.info(
            f"Constructed the delaunay triangulation with furthest_site = False and {len(self.tesselation.simplices)} simplices."
        )

    def predict(self, alpha: float) -> None:
        """
        Creates the Alpha complex by filtering the Delaunay triangulation.

        Parameters
        ----------
        alpha: 
            Parameterises the constructed complex.
        """
        if alpha < 0:
            raise AlphaValueError(
                self.alpha, 
                type(self),
                LOGGER
            )

        if alpha == 0:
            self.alpha_complex = ConvexHull(self.vertices).simplices
            return

        simplices = self.vertices.take(self.tesselation.simplices, axis = 0)
        simplices_circum_radii = self._circum_radius(simplices)
        bool_index = (simplices_circum_radii <= (1 / alpha))
        picked_simplex_indices = np.concatenate([np.where(bool_index)[0]])
        picked_simplices = self.tesselation.simplices[picked_simplex_indices]

        unique_triangles = self._face_filter(picked_simplices)

        unique_edges = self._face_filter(unique_triangles)

        self.alpha_complex = [unique_triangles, unique_edges]

        LOGGER.info(
            f"\u03B1-complex with {self.n_simplices} simplices generated."
        )

    @staticmethod
    def _circum_radius(tessellation_vertices: np.ndarray) -> np.ndarray:
        """
        Calculates the circum radii of simplices by utilising determinants.

        Parameters
        ----------
        tesselation_vertices:
            A numpy ndarray containing the simplices. Usually of the complex (`n, k + 1, k`). This usually denotes that there are `n` simplices each with dimension `k` and number of points `k + 1`.

        Returns
        -------
        ``numpy array``:
            Array containing the circum radii of the `n` simplices.
        """
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
        dem += (dem == 0)
        return np.sqrt(num) / dem

    def get_summary(self, alpha: float, dataset: str) -> pd.DataFrame:
        """
        Produces a summary of the generated complex.

        Parameters
        ----------
        alpha:
            The value used to construct the Alpha complex.

        dataset:
            Name of the dataset on which the complex was constructed.

        Returns
        -------
        ``pandas DataFrame``:
            A dataframe of values.
        """
        summary = {
            "Alpha": alpha,
            "Number of vertices": int(len(self.vertices)),
            "Number of simplices": self.n_simplices,
        }

        return pd.DataFrame.from_dict(
            summary,
            orient = "index",
            columns = [f"Dataset: {dataset}"]
        )

class AlphaComplexND(AlphaComplex):
    """
    A specialised interface that implements `AlphaComplex` used to construct `n` D complexes, where :math:`n > 1`.
    """
    def __init__(self, vertices: np.ndarray) -> None:
        self.vertices = vertices
        self.alpha_complex = None
        self.tesselation = None

    def fit(self) -> None:
        """
        Constructs the Delaunay triangulation of the point cloud.
        """
        # Check the tessellation for degenerate simplices and remove them.
        self.tesselation = Delaunay(self.vertices)
        LOGGER.info(
            f"Constructed the delaunay triangulation with furthest_site = False and {len(self.tesselation.simplices)} simplices."
        )

    def predict(self, alpha: float) -> None:
        """
        Creates the Alpha complex by filtering the Delaunay triangulation.

        Parameters
        ----------
        alpha: 
            Parameterises the constructed complex.
        """
        if alpha < 0:
            raise AlphaValueError(
                alpha, 
                type(self),
                LOGGER
            )

        if alpha == 0:
            self.alpha_complex = ConvexHull(self.vertices).simplices
            return
        one_by_alpha = 1 / alpha
        simplices = self.vertices.take(self.tesselation.simplices, axis = 0)
        simplices_circum_radii = self._circum_radius(simplices)
        bool_index = (simplices_circum_radii <= one_by_alpha)
        picked_simplex_indices = np.concatenate([np.where(bool_index)[0]])
        picked_simplices = self.tesselation.simplices[picked_simplex_indices]
        unique_picked_simplices = np.unique(picked_simplices, axis = 0)
        n = self.vertices.shape[1]
        self.alpha_complex = [unique_picked_simplices]
        for _ in range(n, 1, -1):
            unique_picked_simplices = self._face_filter(unique_picked_simplices)
            self.alpha_complex.append(unique_picked_simplices)
        
        LOGGER.info(
            f"\u03B1-complex with {self.n_simplices} simplices generated."
        )
        

    @staticmethod
    def _circum_radius(tessellation_vertices: np.ndarray) -> np.ndarray:
        """
        Calculates the circum radii of simplices by utilising determinants.

        Parameters
        ----------
        tesselation_vertices:
            A numpy ndarray containing the simplices. Usually of the complex (`n, k + 1, k`). This usually denotes that there are `n` simplices each with dimension `k` and number of points `k + 1`.

        Returns
        -------
        ``numpy array``:
            Array containing the circum radii of the `n` simplices.
        """
        def _circum_radius_helper(tessellation_vertices: np.ndarray):
            return [
                np.linalg.norm(simplex[:, None] - simplex, axis = -1) ** 2
                for simplex in tessellation_vertices
            ]

        distances = _circum_radius_helper(tessellation_vertices)
        circum_radii = list()
        for dm in distances:
            m, n = dm.shape
            cm = np.block([
                [0,               np.ones((1, n))],
                [np.ones((m, 1)), dm             ],
            ])
            circum_radii.append(
                np.sqrt(np.linalg.det(dm) / (-2 * np.linalg.det(cm)))
            )
        return np.array(circum_radii, dtype = np.float64)
            
    def get_summary(self, alpha: float, dataset: str) -> pd.DataFrame:
        """
        Produces a summary of the generated complex.

        Parameters
        ----------
        alpha:
            The value used to construct the Alpha complex.

        dataset:
            Name of the dataset on which the complex was constructed.

        Returns
        -------
        ``pandas DataFrame``:
            A dataframe of values.
        """
        summary = {
            "Alpha": alpha,
            "Number of vertices": int(len(self.vertices)),
            "Number of simplices": self.n_simplices,
        }

        return pd.DataFrame.from_dict(
            summary,
            orient = "index",   
            columns = [f"Dataset: {dataset}"],
        )