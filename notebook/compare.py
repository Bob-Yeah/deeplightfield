import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(sys.path[0] + '/../'))

from utils import img


def plot_compares(dataset, compares, view_range):
    gt_exist = os.path.exists(dataset)
    fig_cols = len(compares) + int(gt_exist)

    def load_perf(model, epoch):
        for file in os.listdir(model):
            if file.startswith(f'perf_model-epoch_{epoch}_{dataset}'):
                csv_data = np.loadtxt(os.path.join(model, file), dtype=np.str, delimiter=", ")
                return csv_data[1:].astype(np.float32)
        return None
    for title in compares:
        compares[title].append(load_perf(compares[title][0], compares[title][1]))

    for i in view_range:
        plt.figure(facecolor='white', figsize=(4 * fig_cols, 4))
        plt.suptitle('View %d' % i)
        j = 1
        if gt_exist:
            plt.subplot(1, fig_cols, j)
            plt.title('Ground truth')
            img.plot(img.load(f"{dataset}/view_{i:0>4d}.png"))
            j += 1
        for key in compares:
            plt.subplot(1, fig_cols, j)
            title = key
            evals = []
            if compares[key][3] is not None:
                if not np.isnan(compares[key][3][i, 2]):
                    evals.append('%.2f dB' % compares[key][3][i, 2])
                if compares[key][3].shape[1] >= 4 and not np.isnan(compares[key][3][i, 3]):
                    evals.append('SSIM: %.2f' % compares[key][3][i, 3])
            plt.title(title + f" ({', '.join(evals)})" if len(evals) > 0 else '')
            img.plot(img.load(f'{compares[title][0]}/output/model-epoch_{compares[title][1]}/{dataset}/' +
                              compares[title][2] % i))
            j += 1
