from PIL import Image
from math import sqrt


def red_mean(color1, color2):
    r_m = (color1[0] + color2[0])/2
    return sqrt((2+r_m/256)*(color2[0]-color1[0])**2 + 4*(color2[1]-color1[1])**2 + (2+(255-r_m)/256)*(color2[2]-color1[2])**2)


def bounding_box(path: str):
    im = Image.open(path)
    pix = im.load()
    width, height = im.size

    box_left, box_right = width, 0
    box_top, box_bottom = 0, height-1
    prev_plane = False
    for r in range(1, height):
        plane = False
        for c in range(1, width):
            diff_h = red_mean(pix[c-1, r], pix[c, r])
            diff_v = red_mean(pix[c, r-1], pix[c, r])
            if diff_h > 100:
                if c < box_left: box_left = c
                if c > box_right: box_right = c
            if diff_v > 100: plane = True
        if plane and not prev_plane: box_top = r
        if not plane and prev_plane:
            box_bottom = r-1
            if box_bottom - box_top > 50: break
        prev_plane = plane

    return (box_left, box_top), (box_right, box_bottom), (box_bottom - box_top) / (box_right - box_left)
