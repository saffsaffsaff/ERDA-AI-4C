import glob
import numpy as np


def get_specs(file):
    # only use the last part of the image name which gives info about the aircraft
    split_string = file.split("page 1")
    substring = split_string[1]
    # split info about aircraft into parts
    subsubstring = substring.split(" - ")
    sub3string = subsubstring[1].split(' ')
    # remove .jpg or a slash and return
    return subsubstring[0][1:], sub3string[1], subsubstring[-1][:-4], sub3string[0]  # code, type, fleet carrier, manufacturer


if __name__ == '__main__':  # if this file is run
    codes = []
    types = []
    fleet_carriers = []
    manufacturers = []
    i = 0

    # go over the pictures in the database one by one
    for filename in glob.iglob(r'page 1' + '**/*.JPG', recursive=True):
        # make sure it takes 100 pictures and not more
        i += 1
        if i == 101:
            break

        code, aircraft_type, fleet_carrier, manufacturer = get_specs(filename)

        codes.append(code)
        types.append(aircraft_type)
        fleet_carriers.append(fleet_carrier)
        manufacturers.append(manufacturer)

    # convert the lists to arrays
    code_array = np.array(codes)
    type_array = np.array(types)
    fleet_carrier_array = np.array(fleet_carriers)
    manufacturer_array = np.array(manufacturers)

    print('code: ', code_array)
    print('type: ', type_array)
    print('fleet_carrier: ', fleet_carrier_array)
    print('manufacturer: ', manufacturer_array)


    def number_diff_elements(list):
        for i in range(0, 100):
            for i in range(0, 100):
                if i >= len(list):
                    break
                n = list.count(list[i])
                if n > 1:
                    list.remove(list[i])
        return list


    print('number of codes is: ', len(number_diff_elements(codes)))
    print('number of types is: ', len(number_diff_elements(types)))
    print('number of fleetcarriers is: ', len(number_diff_elements(fleet_carriers)))
    print('number of manufacturers is: ', len(number_diff_elements(manufacturers)))

    ## Create Excel Sheet - Appendix with table of aeroplanes
    '''import os
    import pandas as pd
    
    array = [['a1', 'a2', 'a3'],
             ['a4', 'a5', 'a6'],
             ['a7', 'a8', 'a9'],
             ['a10', 'a11', 'a12', 'a13', 'a14']]
    
    df = pd.DataFrame(array).T
    df.to_excel(excel_writer = os.getcwd() + 'appendixa.xlsx')'''