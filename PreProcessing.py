import glob, pandas as pd, numpy as np
from PreProcessing_AspectRatio import bounding_box
from PreProcessing_Colors import color_distribution
from ArrayMaker import get_specs

df = pd.DataFrame()
filenames = list(glob.iglob(r'page 1' + '**/*.jpg', recursive=True))
no_files = len(filenames)
print(no_files)

types, carriers = zip(*[get_specs(file)[1:3] for file in filenames])
types, carriers = list(set(types)), list(set(carriers))

print(len(types), types)
print(len(carriers), carriers)

for i, file in enumerate(filenames):
    specs = get_specs(file)  # get airplane specifications from filename
    type_ind, carrier_ind = types.index(specs[1]), carriers.index(specs[2])
    top_left, bottom_right, ratio = bounding_box(file)  # determine bounding box of airplane
    occurrences, avg_locations, avg_locations_sq = color_distribution(file, top_left, bottom_right)  # determine color distribution

    values = np.concatenate(([type_ind], [carrier_ind], [ratio], occurrences.flatten(), avg_locations.flatten(), avg_locations_sq.flatten()))  # create one large numpy array with all numerical values (1D)
    df = df.append([[*specs[1:3], *values]])  # add properties of current airplane to the DataFrame

    print(int((i+1)/no_files*100), '%')  # print progress

df.to_pickle('./processed_data.pkl')  # export DataFrame
