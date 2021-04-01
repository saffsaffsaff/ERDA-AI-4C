from PIL import Image
import numpy as np


def color_distribution(path, top_left, bottom_right):
    width, height = bottom_right[0] - top_left[0] + 1, bottom_right[1] - top_left[1] + 1
    no_pixels = width * height  # number of pixels

    pix = Image.open(path).load()  # load image pixels
    occurrences = np.zeros((5, 5, 5))
    avg_locations = np.zeros((5, 5, 5, 2))
    avg_locations_sq = np.zeros((5, 5, 5, 2))
    for r in range(top_left[1], bottom_right[1]+1):
        for c in range(top_left[0], bottom_right[0]+1):
            color = (int(pix[c, r][0]/51.2), int(pix[c, r][1]/51.2), int(pix[c, r][2]/51.2))  # from (0-255, 0-255, 0-255) to (0-4, 0-4, 0-4), i.e., 125 different colors
            occurrences[color[0]][color[1]][color[2]] += 1 / no_pixels  # update the percentage of color occurrences
            no_color = occurrences[color[0]][color[1]][color[2]] * no_pixels  # current total number of pixels with this color
            avg_locations[color[0]][color[1]][color[2]][0] = avg_locations[color[0]][color[1]][color[2]][0] * (no_color - 1) / no_color + (c - top_left[0] + 0.5) / width / no_color  # iterative average of x-coordinate
            avg_locations[color[0]][color[1]][color[2]][1] = avg_locations[color[0]][color[1]][color[2]][1] * (no_color - 1) / no_color + (r - top_left[0] + 0.5) / height / no_color  # iterative average of y-coordinate
            avg_locations_sq[color[0]][color[1]][color[2]][0] = avg_locations_sq[color[0]][color[1]][color[2]][0] * (no_color - 1) / no_color + ((c - top_left[0] + 0.5) / width) ** 2 / no_color  # iterative average of x-coordinate squared
            avg_locations_sq[color[0]][color[1]][color[2]][1] = avg_locations_sq[color[0]][color[1]][color[2]][1] * (no_color - 1) / no_color + ((r - top_left[0] + 0.5) / height) ** 2 / no_color  # iterative average of y-coordinate squared

    return occurrences, avg_locations, avg_locations_sq
