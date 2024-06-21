import os

from matplotlib import pyplot as plt

img_path = './data/DoReMi_v1/measure_cut/Images/'
max_h, max_w = 0, 0

for image in os.listdir(img_path):
    # open image and get dim
    img = plt.imread(f"{img_path}/{image}")
    w, h = img.shape[:2]
    max_h = max(max_h, h)
    max_w = max(max_w, w)

print(max_h, max_w)