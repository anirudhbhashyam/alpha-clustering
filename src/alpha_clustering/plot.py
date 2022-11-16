from dataclasses import dataclass, field

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

from .logger import Logger

LOGGER = Logger(__name__)

COLOR_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
    "#ffffff", "#000000", "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff", "#ff00ff", "#800000", "#800080",
    "#008000", "#008080", "#000080", "#000000", "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff", "#ff00ff",
    "#800000", "#800080", "#008000", "#008080", "#000080", "#000000", "#ff0000", "#00ff00", "#0000ff", "#ffff00",
]

plt.rcParams["axes.edgecolor"] = (0.33, 0.32, 0.29)
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.color"] = (0.33, 0.32, 0.29)
plt.rcParams["ytick.color"] = (0.33, 0.32, 0.29)

@dataclass
class Plot:
    vertices: np.ndarray
    show_q: bool = False
    dimension: int = field(
        init = False
    )

    def __post_init__(self):
        if len(self.vertices.shape) != 2:
            raise ValueError("The vertices must be a 2D array of points.")
        self.dimension = self.vertices.shape[1]

    def alpha_shape(
        self,
        shape: np.ndarray,
        figsize: tuple[float, float] = (16, 9),
        points_q: bool = True,
        ticks_q: bool = True,
    ) -> plt.Figure:
        LOGGER.info("Visualizing \u03B1-shape...")

        
        # Find the triangles in the shape.
        for simplices in shape:
            if simplices.shape[1] == 3:
                triangles = simplices
                break

        fig, ax = self._init_fig(figsize)

        if points_q:
            _ = self.points_scatter(ax)

        if self.dimension == 2:
            ax.triplot(
                *self.vertices.T, 
                triangles = triangles, 
                linewidth = 1.0, 
                color = sns.color_palette("mako", 50)[20]
            )

        if self.dimension >= 3:
            ax.plot_trisurf(
                *(self.vertices[:, : 3]).T,
                triangles = triangles, 
                linewidth = 1.0, 
                color = sns.color_palette("mako", 50)[20]
            )
            
        self._post_process_plot(ax, ticks_q)

        return fig

    def clusters(
        self, 
        clusters: np.array,
        figsize: tuple[float, float] = (16, 9),
        ticks_q: bool = True
    ) -> plt.Figure:

        LOGGER.info("Visualizing clusters...")

        fig, ax = self._init_fig(figsize)

        NUM_COLORS = len(clusters)

        cm = plt.get_cmap("rainbow")
        ax.set_prop_cycle("color", [cm(i / NUM_COLORS) for i in range(NUM_COLORS)])

        for label, c in enumerate(clusters):
            ax.scatter(
                *self.vertices[list(c)].T,
                label = f"Cluster {label}",
                alpha = 0.8,
                linewidth = 0.0
            )

        # ax.legend(loc = "upper right")

        self._post_process_plot(ax, ticks_q)

        return fig

    def points_scatter(
        self,
        ax: plt.Axes = None,
        clusters: np.array = None
    ) -> plt.Figure | PathCollection:

        if clusters is None:
            clusters = [sns.color_palette("mako", 50)[30]]

        if ax is None: 
            fig, ax = self._init_fig()
            ax.scatter(
                *self.vertices.T,
                c = clusters,
                linewidth = 0.0,
                alpha = 0.8,
                cmap = COLOR_PALETTE
            )
            self._post_process_plot(ax, True)
            return fig

        return ax.scatter(
            *self.vertices.T,
            c = clusters,
            linewidth = 0.0,
            alpha = 0.8,
            cmap = COLOR_PALETTE
        )

    def _post_process_plot(
        self,
        ax: plt.Axes,
        ticks_q: bool,
        legend_elements: tuple[list, list] = None
    ) -> None:
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if not ticks_q:
            ax.set_xticks([])
            ax.set_yticks([])
            if self.dimension == 3:
                ax.set_zticks([])
        if legend_elements is not None:
            legends, labels = legend_elements
            ax.legend(
                handles = legends,
                labels = labels
            )
        if self.show_q:
            plt.show()

    def _init_fig(self, figsize: tuple[float, float] = (16, 9)) -> \
    tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111, projection = "3d" if self.dimension == 3 else None)
        return fig, ax