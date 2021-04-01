from PIL import Image
import numpy as np


def color_distribution(path, top_left, bottom_right):
    width, height = bottom_right[0] - top_left[0] + 1, bottom_right[1] - top_left[1] + 1
    no_pixels = width * height

    pix = Image.open(path).load()
    occurrences = np.zeros((3, 256))
    avg_locations = np.zeros((3, 256, 2))
    avg_locations_sq = np.zeros((3, 256, 2))
    for r in range(top_left[1], bottom_right[1]+1):
        for c in range(top_left[0], bottom_right[0]+1):
            for i in range(3):
                occurrences[i][pix[c, r][i]] += 1/no_pixels
                no_color = occurrences[i][pix[c, r][i]] * no_pixels  # current total number of pixels with this shade
                avg_locations[i][pix[c, r][i]][0] = avg_locations[i][pix[c, r][i]][0] * (no_color - 1) / no_color + (c - top_left[0]) / (width-1) / no_color  # iterative average of x-coordinate
                avg_locations[i][pix[c, r][i]][1] = avg_locations[i][pix[c, r][i]][1] * (no_color - 1) / no_color + (r - top_left[1]) / (height-1) / no_color  # iterative average of y-coordinate
                avg_locations_sq[i][pix[c, r][i]][0] = avg_locations_sq[i][pix[c, r][i]][0] * (no_color - 1) / no_color + ((c - top_left[0]) / (width-1))**2 / no_color  # iterative average of x-coordinate squared
                avg_locations_sq[i][pix[c, r][i]][1] = avg_locations_sq[i][pix[c, r][i]][1] * (no_color - 1) / no_color + ((r - top_left[1]) / (height-1))**2 / no_color  # iterative average of y-coordinate squared

    return occurrences, avg_locations, avg_locations_sq
