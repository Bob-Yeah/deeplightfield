import sys
import os
import argparse
import torch
import torch.optim
from torch import onnx

sys.path.append(os.path.abspath(sys.path[0] + '/../'))
__package__ = "deep_view_syn"

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

from .msl_net import MslNet
from .configs.spherical_view_syn import SphericalViewSynConfig
from .my import device
from .my import netio
from .my import util

dir_path, model_file = os.path.split(opt.model)
batch_size = eval(opt.batch_size)
os.chdir(dir_path)

config = SphericalViewSynConfig()

def load_net(path):
    name = os.path.splitext(os.path.basename(path))[0]
    config.from_id(name)
    config.SAMPLE_PARAMS['spherical'] = True
    config.SAMPLE_PARAMS['perturb_sample'] = False
    config.SAMPLE_PARAMS['n_samples'] = 4
    config.print()
    net = MslNet(config.FC_PARAMS, config.SAMPLE_PARAMS, config.GRAY,
                 config.N_ENCODE_DIM, export_mode=True).to(device.GetDevice())
    netio.LoadNet(path, net)
    return net, name


if __name__ == "__main__":
    with torch.no_grad():
        # Load model
        net, name = load_net(model_file)

        # Input to the model
        rays_o = torch.empty(batch_size, 3, device=device.GetDevice())
        rays_d = torch.empty(batch_size, 3, device=device.GetDevice())

        util.CreateDirIfNeed(opt.outdir)

        # Export the model
        outpath = os.path.join(opt.outdir, config.to_id() + ".onnx")
        onnx.export(
            net,                 # model being run
            (rays_o, rays_d),    # model input (or a tuple for multiple inputs)
            outpath,
            export_params=True,  # store the trained parameter weights inside the model file
            verbose=True,
            opset_version=9,                 # the ONNX version to export the model to
            do_constant_folding=True,        # whether to execute constant folding
            input_names=['Rays_o', 'Rays_d'],  # the model's input names
            output_names=['Colors']  # the model's output names
        )
        print ('Model exported to ' + outpath)
