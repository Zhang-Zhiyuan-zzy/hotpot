"""
python v3.9.0
@Project: hotpot
@File   : defaults
@Auther : Zhiyuan Zhang
@Data   : 2024/10/14
@Time   : 16:02
"""
class Settings:
    # General
    font = 'Arial'

    # Figure
    figwidth = 10.72  # inch
    figheight = 8.205  # inch
    dpi = 300

    # Axes
    axes_pos = [0.1483, 0.1657, 0.7163, 0.7185]
    splinewidth = 3
    xy_label_fontsize = 32
    ticklabels_fontsize = 22
    xticklabels_fontsize = ticklabels_fontsize
    yticklabels_fontsize = ticklabels_fontsize
    ticklabels_rotation = None
    xticklabels_rotation = None
    yticklabels_rotation = None


    superscript_position = (-0.125, 1.075)
    superscript_font_dict = {'font': font, 'fontsize': 42, "fontweight": 'bold'}

    # Colorbar
    colorbar_pos = [0.90, 0.1657, 0.05, 0.7185]

    # Insert text
    text_fontdict = {'fontsize': 20}
