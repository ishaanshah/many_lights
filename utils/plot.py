import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from utils.render import linear_to_srgb
from typing import Any, Dict, List

def convert_to_red_green(img_list: List[Any]) -> np.ndarray:
    results = []
    for img in img_list:
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        result = np.zeros_like(img)
        result[:,:,0] = -2 * np.mean(np.minimum(img, np.zeros(3)), axis=2)
        result[:,:,1] = 2 * np.mean(np.maximum(img, np.zeros(3)), axis=2)
        results.append(result)
    return results

# Helper function used for visualization in the following examples
def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)

def plot_line(
    x: Any, y: Any,
    title: str="",
    xlabel: str="x",
    ylabel: str="y",
    xticks: list=[],
    legend: list=[],
    **kwargs
):
    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    ax.plot(x, y, **kwargs)
    ax.set_title(title)
    if legend:
        ax.legend(legend, loc="upper right")
    if xticks:
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax

def plot_img_grid(
    img_grid: List[List[Any]],
    widths: List[int]|int,
    heights: List[int]|int,
    x_labels: List[str]=[],
    y_labels: List[str]=[],
    title: str="",
    width_inch: int=8,
    srgb_rows: List[int]|"all"= [],
    **kwargs
):
    nrows = len(img_grid)
    ncols = len(img_grid[0])

    if isinstance(widths, list):
        total_width = sum(widths)
    else:
        total_width = widths * ncols

    if isinstance(heights, list):
        total_height = sum(heights)
    else:
        total_height = heights * nrows

    figsize = (width_inch, total_height / total_width * width_inch)

    fig = plt.figure(figsize=figsize)
    ax = fig.subplots(nrows=nrows, ncols=ncols)
    for row in range(nrows):
        for col in range(ncols):
            if row in srgb_rows or srgb_rows == "all":
                ax[row][col].imshow(linear_to_srgb(img_grid[row][col]), **kwargs)
            else:
                ax[row][col].imshow(img_grid[row][col], **kwargs)
            
            # Set titles
            if col == 0 and y_labels:
                ax[row][col].set_ylabel(y_labels[row])
            if row == len(img_grid) - 1 and x_labels:
                ax[row][col].set_xlabel(x_labels[col])

            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])

    fig.suptitle(title)
    return fig

def plot_img_dict(
    imgs: list,
    title: str="",
    srgb: bool=True,
    **kwargs
):
    fig = plt.figure(layout="constrained")
    
    # Create mosiac
    mosaic = []
    dict_ = {}
    for row in imgs:
        if isinstance(row, list):
            row_keys = []
            for col in row:
                key, val = list(col.items())[0]
                row_keys.append(key)
                dict_[key] = val
            mosaic.append(row_keys)
        elif isinstance(row, dict):
            key, val = list(row.items())[0]
            mosaic.append(key)
            dict_[key] = val
        else:
            raise ValueError

    ax_dict: Dict[str, plt.Axes] = fig.subplot_mosaic(mosaic, **kwargs)
    for key, val in dict_.items():
        if srgb:
            ax_dict[key].imshow(linear_to_srgb(val))
        else:
            ax_dict[key].imshow(val)
        ax_dict[key].set_axis_off()
    fig.suptitle(title)

    return fig

def plot_polar(
    theta: Any, phi: Any, f: Any,
    title: str="",
    full=False,    # TODO: Allow plotting entire sphere
    axes: plt.Axes=None,
    **kwargs
):
    if axes is None:
        fig = plt.figure()
        axes: plt.Axes = fig.add_subplot(projection="polar")

    axes.contourf(phi, 1-np.cos(theta), np.asarray(f).reshape(len(theta), len(phi)).T, **kwargs)
    axes.grid(False)
    axes.set_yticks([])
    axes.set_title(title)

    if axes is None:
        return fig