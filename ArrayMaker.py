import glob
import numpy as np

codes = []
types = []
fleet_carriers = []
manufacturers = []
i = 0

for filename in glob.iglob(r'page 1' + '**/*.jpg', recursive=True):
    i += 1
    if i == 101:
        break
    #only use the last part of the image name which gives info about aircraft
    split_string = filename.split("page 1")
    substring = split_string[1]
    #split info about aircraft into the 4 parts
    subsubstring = substring.split(" - ")
    sub3string = subsubstring[1].split(' ')
    #add each information part to a list and remove .jpg or a slash
    codes.append(subsubstring[0][1:])
    types.append(sub3string[1])
    fleet_carriers.append(subsubstring[-1][:-4])
    manufacturers.append(sub3string[0])

#convert the lists to arrays
code_array = np.array(codes)
type_array = np.array(types)
fleet_carrier_array = np.array(fleet_carriers)
manufacturer_array = np.array(manufacturers)

#print('code: ', code_array)
print('type: ', type_array)
print('number of types: ', len(tuple(type_array)))
#print('fleet_carrier: ', fleet_carrier_array)
#print('manufacturer: ', manufacturer_array)