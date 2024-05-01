from math import log2, sqrt
from typing import NoReturn
import numpy as np
import pywt
import argparse
import matplotlib
from PIL import Image
from hilbertcurve.hilbertcurve import HilbertCurve


def panic(message: str) -> NoReturn:
    print(message)
    quit(1)

def main():
    # parse args
    parser = argparse.ArgumentParser(description="Binary file visualization creator")
    parser.add_argument("file", type=str)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument(
        "-g", "--grid-method", choices=["rows", "hilbert"], default="hilbert"
    )
    parser.add_argument(
        "-w", "--wavelet", choices=pywt.wavelist(kind="discrete"), default="haar"
    )
    parser.add_argument(
        "-m",
        "--extrapolation-mode",
        choices=[
            "antireflect",
            "constant",
            "modes",
            "periodic",
            "reflect",
            "symmetric",
            "antisymmetric",
            "msg",
            "periodization",
            "smooth",
            "zero",
        ],
        default="periodic",
    )
    parser.add_argument("-l", "--level", type=int, default=2)
    parser.add_argument(
        "-c", "--colormap", choices=matplotlib.colormaps.keys(), default="inferno"
    )
    args = parser.parse_args()

    # read in file from disk
    f_bytes = []

    with open(args.file, "rb") as file:
        f_bytes = file.read()

    closest_power_of_2 = int(2 ** int(log2(sqrt(len(f_bytes)))))
    p = int(log2(closest_power_of_2))
    hc = HilbertCurve(p=p, n=2, n_procs=3)

    # process file
    # put bytes into a grid:
    f_bytes = np.array(list(f_bytes), dtype=np.uint8)
    match args.grid_method:
        case "rows":
            bytes_2d = f_bytes[:closest_power_of_2**2].reshape((closest_power_of_2,closest_power_of_2))
        case "hilbert":
            indices = hc.distances_from_points(list(np.ndindex((closest_power_of_2,closest_power_of_2))))
            bytes_2d = f_bytes[np.array(indices).reshape((closest_power_of_2,closest_power_of_2))]
        case _:
            panic(f"Grid Method {args.grid_method} not recognized!")

    # apply a wavelet transform:
    wp = pywt.WaveletPacket2D(data=bytes_2d, wavelet=args.wavelet, mode=args.extrapolation_mode)

    image = np.zeros((closest_power_of_2, closest_power_of_2), dtype=np.float32)
    # write wavelet components
    for level in range(1,args.level+1):
        nodes = wp.get_level(level)[:4]
        size = int(closest_power_of_2 / (2**level))
        h = nodes[1]
        v = nodes[2]
        d = nodes[3]
        h_max = np.max(h.data)
        h_min = np.min(h.data)
        image[size:size*2,:size] = ((h.data - h_min) / (h_max - h_min))[:size,:size]
        v_max = np.max(v.data)
        v_min = np.min(v.data)
        image[:size,size:size*2] = ((v.data - v_min) / (v_max - v_min))[:size,:size]
        d_max = np.max(d.data)
        d_min = np.min(d.data)
        image[size:size*2,size:size*2] = ((d.data - d_min) / (d_max - d_min))[:size,:size]

    # write Approximation component
    size = int(closest_power_of_2 / (2**args.level))
    a = wp.get_level(args.level)[0]
    a_max = np.max(a.data)
    a_min = np.min(a.data)
    image[:size,:size] = ((a.data - a_min) / (a_max - a_min))[:size,:size]

    # write image to disk
    image_file = Image.frombytes('RGBA', (closest_power_of_2,closest_power_of_2), (matplotlib.colormaps[args.colormap](image.flatten()) * 255).astype(np.uint8).tobytes(), 'raw')
    image_file = image_file.resize((image_file.width*4, image_file.height*4), resample=Image.Resampling.NEAREST)
    image_file.save(f'{args.output}.png')
    
    # write matrix to disk
    np.savetxt(f'{args.output}.txt', image)

if __name__ == '__main__':
    main()
