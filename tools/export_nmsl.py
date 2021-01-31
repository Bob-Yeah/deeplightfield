import sys
import os
import argparse
import torch
import torch.optim
from torch import onnx
from typing import Mapping, List

sys.path.append(os.path.abspath(sys.path[0] + '/../../'))
__package__ = "deep_view_syn.tools"

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0,
                    help='Which CUDA device to use.')
parser.add_argument('--batch-size', type=str,
                    help='Resolution')
parser.add_argument('--outdir', type=str, default='./',
                    help='Output directory')
parser.add_argument('model', type=str,
                    help='Path of model to export')
opt = parser.parse_args()

# Select device
torch.cuda.set_device(opt.device)
print("Set CUDA:%d as current device." % torch.cuda.current_device())

from ..nets.msl_net_new_export import *
from ..my import util
from ..my import netio
from ..my import device
from ..configs.spherical_view_syn import SphericalViewSynConfig

dir_path, model_file = os.path.split(opt.model)
batch_size = eval(opt.batch_size)
os.chdir(dir_path)

config = SphericalViewSynConfig()


def load_net(path):
    id=os.path.splitext(os.path.basename(path))[0]
    config.from_id(id)
    config.SAMPLE_PARAMS['perturb_sample'] = False
    batch_size_str: str = opt.batch_size.replace('*', 'x')
    config.name += batch_size_str
    config.print()
    net = config.create_net().to(device.GetDevice())
    netio.LoadNet(path, net)
    return net, id


def export_net(net: torch.nn.Module, name: str,
               input: Mapping[str, List[int]], output_names: List[str]):
    outpath = os.path.join(opt.outdir, config.to_id(), name + ".onnx")
    input_tensors = tuple([
        torch.empty(size, device=device.GetDevice())
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
        # Load model`
        net, name = load_net(model_file)

        util.CreateDirIfNeed(os.path.join(opt.outdir, config.to_id()))

        # Export Sampler
        export_net(Sampler(net), 'sampler', {
            'Rays_o': [batch_size, 3],
            'Rays_d': [batch_size, 3]
        }, ['Encoded', 'Depths'])

        # Export FcNet1
        export_net(FcNet1(net), 'fc1', {
            'Encoded': [batch_size, net.n_samples, net.input_encoder.out_dim]
        }, ['Raw'])

        # Export FcNet2
        export_net(FcNet2(net), 'fc2', {
            'Encoded': [batch_size, net.n_samples, net.input_encoder.out_dim]
        }, ['Raw'])

        # Export Cat
        export_net(CatNet(net), 'cat', {
            'Raw1': [batch_size, net.n_samples // 2, 4],
            'Raw2': [batch_size, net.n_samples // 2, 4],
            'Depths': [batch_size, net.n_samples]
        }, ['Colors'])