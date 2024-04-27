from math import log2, sqrt
from typing import NoReturn
import numpy as np
import pywt
import argparse
from coloraide import Color
from PIL import Image


def panic(message: str) -> NoReturn:
    print(message)
    quit(1)

# parse args

parser = argparse.ArgumentParser(description='Binary file visualization creator')
parser.add_argument('file', type=str)
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-g', '--grid-method', type=str, default="rows")
parser.add_argument('-w', '--wavelet', type=str, default="haar")
parser.add_argument('-l', '--level', type=int, default=1)
args = parser.parse_args()

# read in file from disk
f_bytes = []

with open(args.file, 'rb') as file:
    f_bytes = file.read()

closest_power_of_2 = int(2**int(log2(sqrt(len(f_bytes)))))

# process file
# put bytes into a grid:
bytes_2d = np.zeros((closest_power_of_2, closest_power_of_2))

for i in range(closest_power_of_2):
    for j in range(closest_power_of_2):
        match args.grid_method:
            case "rows":
                index_1d = i + j * closest_power_of_2
            case _:
                panic(f"Grid Method {args.grid_method} not recognized!")
        bytes_2d[i,j] = f_bytes[index_1d]


# apply a wavelet transform:
wp = pywt.WaveletPacket2D(data=bytes_2d, wavelet=args.wavelet, mode='symmetric')

color_interp = Color.interpolate(["rgb(0,0,4)", "rgb(40,11,84)", "rgb(101,21,110)", "rgb(159,42,99)", "rgb(212,72,66)", "rgb(245,125,21)", "rgb(250,193,39)", "rgb(252,255,164)"], space='oklab')
def get_wavelet_color(value, i_max, i_min):
    c = color_interp((value - i_min) / (i_max-i_min))
    return ([int(255 * min(max(_, 0), 1)) for _ in c.convert('srgb').coords()])

image = np.zeros((closest_power_of_2, closest_power_of_2, 3), dtype=np.uint8)
# write wavelet components
for level in range(1,args.level+1):
    nodes = wp.get_level(level)[:4]
    size = int(closest_power_of_2 / (2**level))
    h = nodes[1]
    v = nodes[2]
    d = nodes[3]
    h_max = np.max(h.data)
    h_min = np.min(h.data)
    for i in range(size):
        for j in range(size):
            image[size+i, j] = get_wavelet_color(h.data[i,j], h_max, h_min)
    v_max = np.max(v.data)
    v_min = np.min(v.data)
    for i in range(size):
        for j in range(size):
            image[i, size+j] = get_wavelet_color(v.data[i,j], v_max, v_min)
    d_max = np.max(d.data)
    d_min = np.min(d.data)
    for i in range(size):
        for j in range(size):
            image[size+i, size+j] = get_wavelet_color(d.data[i,j], d_max, d_min)

# write Approximation component
size = int(closest_power_of_2 / (2**args.level))
a = wp.get_level(args.level)[0]
a_max = np.max(a.data)
a_min = np.min(a.data)
for i in range(size):
    for j in range(size):
        image[i, j] = get_wavelet_color(a.data[i,j], a_max, a_min)

image_file = Image.frombytes('RGB', (closest_power_of_2,closest_power_of_2), image.flatten().tobytes(), 'raw')
image_file = image_file.resize((image_file.width*4, image_file.height*4), resample=Image.Resampling.NEAREST)
image_file.save(args.output)

# nodes = wp.get_level(args.level)
# data = [n.data for n in nodes]
#
# normalized_data = np.array([d / np.max(d) for d in data])
# square_len = int(sqrt(normalized_data.shape[0]))
# internal_size = normalized_data.shape[1]
# normalized_data = normalized_data.reshape((square_len, square_len, internal_size, internal_size)).transpose((0,2,1,3)).reshape((square_len*internal_size, square_len*internal_size))
# # wavelet = pywt.Wavelet(args.wavelet)
# # coeffs2 = pywt.dwt2(bytes_2d, args.wavelet)
# # LL, (LH, HL, HH) = coeffs2
#
# # create image
# import matplotlib.pyplot as plt
# plt.imshow(normalized_data, interpolation="nearest", cmap=plt.colormaps['gray'])
# plt.title("multi_level decomposition", fontsize=10)
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()
# plt.show()

# titles = ['Approximation', ' Horizontal detail',
#           'Vertical detail', 'Diagonal detail']
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate([LL, LH, HL, HH]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.colormaps['gray'])
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])
#
# fig.tight_layout()
# plt.show()

# write image to disk
