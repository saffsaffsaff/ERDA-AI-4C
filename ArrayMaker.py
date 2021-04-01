import glob
import numpy as np

codes = []
types = []
fleet_carriers = []
manufacturers = []
i = 0

#go over the pictures in the database one by one
for filename in glob.iglob(r'page 1' + '**/*.JPG', recursive=True):
    #make sure it takes 100 pictures and not more
    i += 1
    if i == 101:
        break
    #only use the last part of the image name which gives info about the aircraft
    split_string = filename.split("page 1")
    substring = split_string[1]
    #split info about aircraft into parts
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

print('code: ', code_array)
print('type: ', type_array)
print('fleet_carrier: ', fleet_carrier_array)
print('manufacturer: ', manufacturer_array)

for i in range(0, 20):
    for i in range(0, 100):
        if i >= len(types):
            break
        n = types.count(types[i])
        if n > 1:
            types.remove(types[i])

print('number of types is: ', len(types))
print(types)

''''## Create Excel Sheet - Appendix with table of aeroplanes
import os
import pandas as pd

array = [['a1', 'a2', 'a3'],
         ['a4', 'a5', 'a6'],
         ['a7', 'a8', 'a9'],
         ['a10', 'a11', 'a12', 'a13', 'a14']]

df = pd.DataFrame(array).T
df.to_excel(excel_writer = os.getcwd() + 'appendixa.xlsx')'''