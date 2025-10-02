#!/usr/bin/env python

############# Required Packages ############
import matplotlib.pyplot as plt, scienceplots, matplotlib.colors as mcolors, os
import matplotlib as mpl

############# Set LaTeX for text rendering ############
mpl.rcParams['text.usetex'] = True

############# PLOT SETTINGS ############
plot_size           = (4, 3)
colors              = ['#e41a1c', '#008000', '#377eb8', '#ff7f00', '#984ea3', '#a65628', '#f781bf']
markers             = ['o', 's', '^', 'D', 'h', 'v', 'p', '*', 'X', '<', '>', '8', 'P', '|', '_']
color_18            = [
    '#008000',  # green
    '#ff7f00',  # orange
    '#984ea3',  # purple
    '#a65628',  # brown
    '#f781bf',  # pink
    "#05545a",  # yellow
    '#4daf4a',  # light green
    '#ffb300',  # gold
    '#1f78b4',  # teal-ish (not blue)
    '#b15928',  # dark brown
    '#999999',  # gray
    "#570F0F",  # Dark Brown
    '#377eb8',  # blue
    '#dede00',  # light yellow
    '#a6cee3',  # light blue
    '#fdbf6f',  # light orange
    '#cab2d6',  # light purple
    '#ffff99',  # light greenish-yellow,
]

color_10            = [
    '#7b3294', 
    '#008837',
    '#ffb300',  # gold
    '#1f78b4',  # teal-ish (not blue)
    "#570F0F",  # Dark Brown
]

color_8             = [
    "#e41a1d",
    "#0b57f9",
    "#03fdf3",
    '#984ea3',
    '#377eb8',
]

dual_colors         = [
    ("#06fb0b", "#008000"),   # green (face, edge)
    ("#56abff", "#0000FF"),   # blue  (face, edge)
    ("#FF5656", "#FF0000"),   # red   (face, edge)
]

graphic_font        = 'sans-serif'
math_font           = 'dejavusans'  # ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']
spine_width         = 1
markersize          = 4
capsize             = 3
markeredgewidth     = 0.75
legend_linewidth    = 1
linewidth           = 1
tick_width          = 0.75
tick_length         = 4
minor_tick_width    = 0.5
minor_tick_length   = 2
tick_labelsize      = 10
legend_fontsize     = 8
legend_boxwidth     = 0.75
label_fontsize      = 12
borderaxespad       = 0.6
alpha               = 0.5
resolution_value    = 1200

############# FUNCTION TO PROCESS COLORS ############
def face_colors(colors, alpha):
    rgba_colors = [mcolors.to_rgba(c) for c in colors]
    
    return [(rgba[0], rgba[1], rgba[2], alpha) for rgba in rgba_colors]

face_colors = face_colors(colors, alpha)

####################### PLOT FUNCTIONS ##############

def plot_init():
    
    """Creates a matplotlib figure and axis with predefined styles and applies tick settings."""
    
    with plt.style.context(['ieee']):
        plt.rcParams['font.family'] = graphic_font
        plt.rcParams['mathtext.fontset'] = math_font
        
        fig, ax = plt.subplots(figsize=plot_size)

        # Set spine widths
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)

        # Apply tick parameters
        ax.tick_params(axis='both', which='major', direction='in', width=tick_width, length=tick_length,
                       labelsize=tick_labelsize, bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in', width=minor_tick_width, length=minor_tick_length,
                       bottom=True, top=True, left=True, right=True)

        return fig, ax

def style_legend(ax, loc, ncol, borderaxespad, fontsize=None, boxwidth=None, edgecolor='black', frame=True):
    if fontsize is None:
        fontsize = legend_fontsize
    if boxwidth is None:
        boxwidth = legend_boxwidth

    legend = ax.legend(fontsize=fontsize, loc=loc, ncol=ncol, borderaxespad=borderaxespad)

    if frame:
        legend.get_frame().set_linewidth(boxwidth)
        legend.get_frame().set_edgecolor(edgecolor)
    else:
        legend.get_frame().set_visible(False)

def save_figure(fig, filename):
    """Saves the figure with the predefined resolution."""
    output_dir = os.getcwd()
    file_path = os.path.join(output_dir, filename)
    fig.savefig(file_path, dpi=resolution_value, bbox_inches='tight')
    fig.savefig(fr"{filename}", dpi=resolution_value, bbox_inches='tight')

#################### END OF CODE #####################