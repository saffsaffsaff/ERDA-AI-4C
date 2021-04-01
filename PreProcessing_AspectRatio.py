from PIL import Image
from math import sqrt


def red_mean(color1, color2):  # calculate the difference between two colors (low-cost approximation, a.k.a., 'redmean')
    r_m = (color1[0] + color2[0])/2
    return sqrt((2+r_m/256)*(color2[0]-color1[0])**2 + 4*(color2[1]-color1[1])**2 + (2+(255-r_m)/256)*(color2[2]-color1[2])**2)


def bounding_box(path: str):  # calculate the bounding box around the airplane in the image
    # import image
    im = Image.open(path)
    pix = im.load()
    width, height = im.size

    # initialise bounding box values
    box_left, box_right = width, 0
    box_top, box_bottom = 0, height-1
    prev_plane = False  # holds the previous value of 'plane'
    # loop through all pixels
    for r in range(1, height):  # for each row
        plane = False  # is True when the airplane is detected on the current row, and False otherwise
        for c in range(1, width):  # for each column
            diff_h = red_mean(pix[c-1, r], pix[c, r])  # horizontal color difference
            diff_v = red_mean(pix[c, r-1], pix[c, r])  # vertical color difference
            if diff_h > 100:  # a color difference threshold of 100 seams to work well
                if c < box_left: box_left = c
                if c > box_right: box_right = c
            if diff_v > 100: plane = True
        if plane and not prev_plane: box_top = r  # if the airplane is detected on this row but not the previous one
        if not plane and prev_plane:  # if the airplane was detected on the previous row but not the current one
            box_bottom = r-1
            if box_bottom - box_top > 50: break  # set minimum height of plane
        prev_plane = plane

    return (box_left, box_top), (box_right, box_bottom), (box_bottom - box_top) / (box_right - box_left)
