from math import log2, sqrt
from typing import NoReturn
import numpy as np
import pywt
import argparse

def panic(message: str) -> NoReturn:
    print(message)
    quit(1)

# parse args

parser = argparse.ArgumentParser(description='Binary file visualization creator')
parser.add_argument('file', type=str)
parser.add_argument('--grid-method', type=str, default="rows")
parser.add_argument('--wavelet', type=str, default="haar")
parser.add_argument('--level', type=int, default=1)
args = parser.parse_args()

# read in file from disk
f_bytes = []

with open(args.file, 'rb') as file:
    f_bytes = file.read()

closest_power_of_2 = int(sqrt(2**int(log2(len(f_bytes)))))

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
nodes = wp.get_level(args.level)
data = [n.data for n in nodes]

normalized_data = np.array([d / np.max(d) for d in data])
square_len = int(sqrt(normalized_data.shape[0]))
internal_size = normalized_data.shape[1]
normalized_data = normalized_data.reshape((square_len, square_len, internal_size, internal_size)).transpose((0,2,1,3)).reshape((square_len*internal_size, square_len*internal_size))
# wavelet = pywt.Wavelet(args.wavelet)
# coeffs2 = pywt.dwt2(bytes_2d, args.wavelet)
# LL, (LH, HL, HH) = coeffs2

# create image
import matplotlib.pyplot as plt
plt.imshow(normalized_data, interpolation="nearest", cmap=plt.colormaps['gray'])
plt.title("multi_level decomposition", fontsize=10)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

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
