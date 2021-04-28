import sys
import os
import argparse
import torch
import torch.optim
from torch import onnx
from typing import Mapping, List

sys.path.append(os.path.abspath(sys.path[0] + '/../'))

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0,
                    help='Which CUDA device to use.')
parser.add_argument('--batch-size', type=str,
                    help='Resolution')
parser.add_argument('model', type=str,
                    help='Path of model to export')
opt = parser.parse_args()

# Select device
torch.cuda.set_device(opt.device)
print("Set CUDA:%d as current device." % torch.cuda.current_device())

from nets.snerf_fast import *
from utils import misc
from utils import netio
from utils import device
from configs.spherical_view_syn import SphericalViewSynConfig

dir_path, model_file = os.path.split(opt.model)
batch_size = eval(opt.batch_size)
batch_size_str = opt.batch_size.replace('*', 'x')
outdir = f"output_{int(os.path.splitext(model_file)[0][12:])}"

os.chdir(dir_path)
misc.create_dir(outdir)

config = SphericalViewSynConfig()


def load_net(path):
    id=os.path.split(dir_path)[-1]#os.path.splitext(os.path.basename(path))[0]
    config.from_id(id)
    config.SAMPLE_PARAMS['perturb_sample'] = False
    config.name += batch_size_str
    config.print()
    net = config.create_net().to(device.default())
    netio.load(path, net)
    return net, id


def export_net(net: torch.nn.Module, name: str,
               input: Mapping[str, List[int]], output_names: List[str]):
    outpath = os.path.join(outdir, f"{name}@{batch_size_str}.onnx")
    input_tensors = tuple([
        torch.empty(size, device=device.default())
        for size in input.values()
    ])
    onnx.export(
        net,
        input_tensors,
        outpath,
        export_params=True,  # store the trained parameter weights inside the model file
        verbose=True,
        opset_version=9,     # the ONNX version to export the model to
        do_constant_folding=True, # whether to execute constant folding
        input_names=input.keys(),   # the model's input names
        output_names=output_names # the model's output names
    )
    print('Model exported to ' + outpath)


if __name__ == "__main__":
    with torch.no_grad():
        net: SnerfFast = load_net(model_file)[0]
        export_net(SnerfFastExport(net), 'net', {
            'Encoded': [batch_size, net.n_samples, net.coord_encoder.out_dim],
            'Depths': [batch_size, net.n_samples]
        }, ['Colors'])