import glob, pandas as pd
from PreProcessing_AspectRatio import bounding_box
from PreProcessing_Colors import color_distribution
from ArrayMaker import get_specs

for img in glob.iglob(r'page 1' + '**/*.jpg', recursive=True):
    top_left, bottom_right, ratio = bounding_box(img)
    occurrences, avg_locations, avg_locations_sq = color_distribution(img, top_left, bottom_right)
