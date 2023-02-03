import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Have colormaps separated into categories:
# http://matplotlib.org/examples/color/colormaps_reference.html

# cmaps = [('Perceptually Uniform Sequential',
#           ['viridis', 'inferno', 'plasma', 'magma']),
#          ('Sequential', ['Blues', 'BuGn', 'BuPu',
#                          'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
#                          'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
#                          'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
#          ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
#                              'copper', 'gist_heat', 'gray', 'hot',
#                              'pink', 'spring', 'summer', 'winter']),
#          ('Diverging', ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
#                         'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
#                         'seismic']),
#          ('Qualitative', ['Accent', 'Dark2', 'Paired', 'Pastel1',
#                           'Pastel2', 'Set1', 'Set2', 'Set3']),
#          ('Miscellaneous', ['gist_earth', 'terrain', 'ocean', 'gist_stern',
#                             'brg', 'CMRmap', 'cubehelix',
#                             'gnuplot', 'gnuplot2', 'gist_ncar',
#                             'nipy_spectral', 'jet', 'rainbow',
#                             'gist_rainbow', 'hsv', 'flag', 'prism'])]
#
# nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
# gradient = np.linspace(0, 1, 256)
# gradient = np.vstack((gradient, gradient))
#
#
# def plot_color_gradients(cmap_category, cmap_list):
#     fig, axes = plt.subplots(nrows=nrows)
#     fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
#     axes[0].set_title(cmap_category + ' colormaps', fontsize=14)
#
#     for ax, name in zip(axes, cmap_list):
#         ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
#         pos = list(ax.get_position().bounds)
#         x_text = pos[0] - 0.01
#         y_text = pos[1] + pos[3] / 2.
#         fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
#
#     # Turn off *all* ticks & spines, not just the ones with colormaps.
#     for ax in axes:
#         ax.set_axis_off()


from matplotlib import cm


def get_jet():
    colormap_int = np.zeros((256, 3), np.uint8)

    for i in range(0, 256, 1):
        colormap_int[i, 0] = np.int_(np.round(cm.jet(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.jet(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(cm.jet(i)[2] * 255.0))


    return colormap_int


def get_afmhot():
    colormap_int = np.zeros((256, 3), np.uint8)

    for i in range(0, 256, 1):
        colormap_int[i, 0] = np.int_(np.round(cm.Blues(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.Blues(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(cm.Blues(i)[2] * 255.0))


    return colormap_int


def tensor2gray(x):
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min)
    x = (x * 255).astype('uint8')
    return x


def gray2color(gray_array, color_map):
    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            color_array[i, j] = color_map[gray_array[i, j]]

    # color_image = Image.fromarray(color_array)

    return color_array


def test_gray2color(im_path):
    gray_image = Image.open(im_path).convert("L")

    gray_array = np.array(gray_image)

    plt.figure()
    plt.subplot(211)
    plt.imshow(gray_array, cmap='gray')

    jet_map = get_jet()
    color_jet = gray2color(gray_array, jet_map)
    plt.subplot(212)
    plt.imshow(color_jet)

    plt.show()

    return

# test_gray2color('/home/sofiahuang/Desktop/20120501134553159.jpg')

# for cmap_category, cmap_list in cmaps:
#     plot_color_gradients(cmap_category, cmap_list)
#
# plt.show()


