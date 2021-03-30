from PIL import Image
from math import sqrt

test_image = "page 1\\HZ-AR27 - Boeing 787-10 Dreamliner - Saudi Arabian Airlines.JPG"

im = Image.open(test_image)
pix = im.load()
width, height = im.size


def red_mean(color1, color2):
    r_m = (color1[0] + color2[0])/2
    return sqrt((2+r_m/256)*(color2[0]-color1[0])**2 + 4*(color2[1]-color1[1])**2 + (2+(255-r_m)/256)*(color2[2]-color1[2])**2)


box_left = width
box_right = 0
box_top = 0
box_bottom = height
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
        box_bottom = r-2
        if box_bottom - box_top > 50: break
    prev_plane = plane


for r in range(height):
    pix[box_left, r] = (255, 0, 0)
    pix[box_right, r] = (0, 0, 255)

for c in range(width):
    pix[c, box_top] = (255, 0, 0)
    pix[c, box_bottom] = (0, 0, 255)

im.show()
