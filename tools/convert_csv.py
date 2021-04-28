import sys
import os
import numpy as np

sys.path.append(os.path.abspath(sys.path[0] + '/../'))

from utils import img


def convert(path):
    with open(path, 'r') as fp:
        content = fp.read()
    if content.startswith('#'):
        content = content[2:]
        with open(path, 'w') as fp:
            fp.write(content)
#    csv_data = np.loadtxt(path, dtype=np.str, delimiter=",")
#    data = csv_data[1:].astype(np.float32)
#    data[:, 2] = img.mse2psnr(data[:, 2])
#    np.savetxt(path, data, fmt=['%d', '%.2f', '%.2f'], delimiter=', ', header='View, Time, PSNR')


if __name__ == "__main__":
    for dirpath, dirnames, filenames in os.walk('../data'):
        for filename in filenames:
            if filename.startswith('perf') and filename.endswith('.csv'):
                print(dirpath, filename)
                convert(os.path.join(dirpath, filename))
