"""
python v3.9.0
@Project: hotpot
@File   : colors
@Auther : Zhiyuan Zhang
@Data   : 2024/10/9
@Time   : 12:54
"""
import os
import os.path as osp
from typing import Union, Optional, Literal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


__all__ = [
    "create_saturation_colormap",
    "register_cmap_to_hotpot",
    "load_defined_cmap"
]

def create_saturation_colormap(
        name: str,
        color_rgb_names: Union[str, tuple[int, int, int]],
):
    """
    Creates a matplotlib colormap from a base RGB color by varying saturation.

    Args:
        name: The name of the colormap.
        color_rgb_names: A tuple (R, G, B) between 0 and 1 or str of color names .

    Returns:
        A matplotlib.colors.LinearSegmentedColormap object.
    """
    # Convert color name to
    if isinstance(color_rgb_names, str):
        rgb = mcolors.to_rgb(color_rgb_names)
    else:
        rgb = color_rgb_names
    assert isinstance(rgb, tuple) and len(rgb) == 3

    # Convert RGB to HSV
    hsv_color = mcolors.rgb_to_hsv(rgb)
    hue = hsv_color[0]
    value = hsv_color[2]

    # Create the HSV just change the saturation
    colors_hsv = []
    # 生成从 0 到 1 的 256 个饱和度值
    for sat in np.linspace(0, 1, 256):
        colors_hsv.append((hue, sat, value))

    # Inverse the HSV to RGB
    colors_rgb = mcolors.hsv_to_rgb(colors_hsv)

    # Generate the Linear Color Map
    return mcolors.LinearSegmentedColormap.from_list(name, colors_rgb)


_cmap_dir = osp.join(osp.dirname(__file__), "cmaps")
def register_cmap_to_hotpot(cmap: mcolors.Colormap, name: Optional[str] = True):
    """ Register a colormap to hotpot local fold. """
    if not isinstance(name, str):
        name = cmap.name

    try:
        np.save(osp.join(_cmap_dir, f"{name}.npy"), cmap.colors)
    except AttributeError as e:
        np.save(osp.join(_cmap_dir, f"{name}.npy"), cmap(range(256)))
    except Exception as e:
        raise e


def get_defined_cmap_name():
    return [fime.split('.')[0] for fime in os.listdir(_cmap_dir)]

def load_defined_cmap(cmap_name: str, cmap_type: Literal['linear', 'list'] = 'linear'):
    if cmap_name not in get_defined_cmap_name():
        raise ValueError(f"{cmap_name} not in {get_defined_cmap_name()}")

    colors_values = np.load(osp.join(_cmap_dir, f"{cmap_name}.npy"))

    if cmap_type.lower() == 'linear':
        return mcolors.LinearSegmentedColormap.from_list(cmap_name, colors_values)
    elif cmap_type.lower() == 'list':
        return mcolors.ListedColormap(colors_values, cmap_name)
    else:
        raise ValueError(f"{cmap_type} is not supported.")


def load_cmap(cmap_name: str, cmap_type: Literal['linear', 'list'] = 'linear'):
    if cmap_name in plt.colormaps:
        return plt.colormaps[cmap_name]
    else:
        try:
            return load_defined_cmap(cmap_name, cmap_type)
        except ValueError:
            raise ValueError(f"Unknown colormap name: {cmap_name}, choose from:\n{list(plt.colormaps) + get_defined_cmap_name()}")


if __name__ == "__main__":
    c_m = create_saturation_colormap(
        "BluesSat",
        "blue"
    )

    register_cmap_to_hotpot(c_m)

