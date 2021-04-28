import os
import importlib
from utils.constants import *
from utils import color
from nets.msl_net import MslNet
from nets.msl_net_new import NewMslNet
from nets.msl_ray import MslRay
from nets.msl_fast import MslFast
from nets.snerf_fast import SnerfFast
from nets.cnerf_v3 import CNerf
from nets.nerf import CascadeNerf
from nets.nerf import CascadeNerf2
from nets.nnerf import NNerf
from nets.nerf_depth import NerfDepth
from nets.bg_net import BgNet
from nets.oracle import Oracle


class SphericalViewSynConfig(object):

    def __init__(self):
        self.name = 'default'

        self.COLOR = color.RGB

        # Net parameters
        self.NET_TYPE = 'msl'
        self.N_ENCODE_DIM = 10
        self.N_DIR_ENCODE = None
        self.NORMALIZE = False
        self.DEPTH_REF = False
        self.FC_PARAMS = {
            'nf': 256,
            'n_layers': 8,
            'skips': [],
            'activation': 'relu'
        }
        self.SAMPLE_PARAMS = {
            'spherical': True,
            'depth_range': (1, 50),
            'n_samples': 32,
            'perturb_sample': True,
            'lindisp': True,
            'inverse_r': True,
        }
        self.NERF_FINE_NET_PARAMS = {
            'enable': False,
            'nf': 256,
            'n_layers': 8,
            'additional_samples': 64
        }

    def load(self, path):
        module_name = os.path.splitext(path)[0].replace('/', '.')
        config_module = importlib.import_module(module_name)
        config_module.update_config(self)
        self.name = module_name.split('.')[-1]

    def load_by_name(self, name):
        config_module = importlib.import_module(
            'configs.' + name)
        config_module.update_config(self)
        self.name = name

    def to_id(self):
        net_type_id = f"{self.NET_TYPE}-{color.to_str(self.COLOR)}"
        encode_id = f"_e{self.N_ENCODE_DIM}"
        dir_encode_id = f"_ed{self.N_DIR_ENCODE}" if self.N_DIR_ENCODE else ''
        fc_id = f"_fc{self.FC_PARAMS['nf']}x{self.FC_PARAMS['n_layers']}"
        skip_id = "_skip%s" % ','.join(['%d' % val for val in self.FC_PARAMS['skips']]) \
            if len(self.FC_PARAMS['skips']) > 0 else ""
        act_id = f"_*{self.FC_PARAMS['activation']}" if self.FC_PARAMS['activation'] != 'relu' else ''
        depth_id = "_d%.2f-%.2f" % (self.SAMPLE_PARAMS['depth_range'][0],
                                    self.SAMPLE_PARAMS['depth_range'][1])
        samples_id = f"_s{self.SAMPLE_PARAMS['n_samples']}"
        ffc_id = f"_ffc{self.NERF_FINE_NET_PARAMS['nf']}x{self.NERF_FINE_NET_PARAMS['n_layers']}"
        fsamples_id = f"_fs{self.NERF_FINE_NET_PARAMS['additional_samples']}"
        fine_id = f"{ffc_id}{fsamples_id}" if self.NERF_FINE_NET_PARAMS['enable'] else ''
        neg_flags = '%s%s%s' % (
            'p' if not self.SAMPLE_PARAMS['perturb_sample'] else '',
            'l' if not self.SAMPLE_PARAMS['lindisp'] else '',
            'i' if not self.SAMPLE_PARAMS['inverse_r'] else ''
        )
        neg_flags = '_~' + neg_flags if neg_flags != '' else ''
        pos_flags = '%s%s' % (
            'n' if self.NORMALIZE else '',
            'd' if self.DEPTH_REF else ''
        )
        pos_flags = '_+' + pos_flags if pos_flags != '' else ''
        return "%s@%s%s%s%s%s%s%s%s%s%s%s" % (self.name, net_type_id, encode_id, dir_encode_id,
                                              fc_id, skip_id, act_id,
                                              depth_id, samples_id,
                                              fine_id,
                                              neg_flags, pos_flags)

    def from_id(self, id: str):
        id_splited = id.split('@')
        if len(id_splited) == 2:
            self.name = id_splited[0]
        segs = id_splited[-1].split('_')
        for i, seg in enumerate(segs):
            if seg.startswith('ffc'):  # Full-connected network parameters
                self.NERF_FINE_NET_PARAMS['nf'], self.NERF_FINE_NET_PARAMS['n_layers'] = (
                    int(str) for str in seg[3:].split('x'))
                self.NERF_FINE_NET_PARAMS['enable'] = True
                continue
            if seg.startswith('fs'):  # Number of samples
                try:
                    self.NERF_FINE_NET_PARAMS['additional_samples'] = int(seg[2:])
                    self.NERF_FINE_NET_PARAMS['enable'] = True
                    continue
                except ValueError:
                    pass
            if seg.startswith('fc'):  # Full-connected network parameters
                self.FC_PARAMS['nf'], self.FC_PARAMS['n_layers'] = (
                    int(str) for str in seg[2:].split('x'))
                continue
            if seg.startswith('skip'):  # Skip connection
                self.FC_PARAMS['skips'] = [int(str)
                                           for str in seg[4:].split(',')]
                continue
            if seg.startswith('*'):  # Activation
                self.FC_PARAMS['activation'] = seg[1:]
                continue
            if seg.startswith('ed'):  # Encode direction
                self.N_DIR_ENCODE = int(seg[2:])
                if self.N_DIR_ENCODE == 0:
                    self.N_DIR_ENCODE = None
                continue
            if seg.startswith('e'):  # Encode
                self.N_ENCODE_DIM = int(seg[1:])
                continue
            if seg.startswith('d'):  # Depth range
                try:
                    self.SAMPLE_PARAMS['depth_range'] = tuple(
                        float(str) for str in seg[1:].split('-'))
                    continue
                except ValueError:
                    pass
            if seg.startswith('s'):  # Number of samples
                try:
                    self.SAMPLE_PARAMS['n_samples'] = int(seg[1:])
                    continue
                except ValueError:
                    pass
            if seg.startswith('~'):  # Negative flags
                if seg.find('p') >= 0:
                    self.SAMPLE_PARAMS['perturb_sample'] = False
                if seg.find('l') >= 0:
                    self.SAMPLE_PARAMS['lindisp'] = False
                if seg.find('i') >= 0:
                    self.SAMPLE_PARAMS['inverse_r'] = False
                if seg.find('n') >= 0:
                    self.NORMALIZE = False
                if seg.find('d') >= 0:
                    self.DEPTH_REF = False
                continue
            if seg.startswith('+'):  # Positive flags
                if seg.find('p') >= 0:
                    self.SAMPLE_PARAMS['perturb_sample'] = True
                if seg.find('l') >= 0:
                    self.SAMPLE_PARAMS['lindisp'] = True
                if seg.find('i') >= 0:
                    self.SAMPLE_PARAMS['inverse_r'] = True
                if seg.find('n') >= 0:
                    self.NORMALIZE = True
                if seg.find('d') >= 0:
                    self.DEPTH_REF = True
                continue
            if i == 0:  # NetType
                self.NET_TYPE, color_str = seg.split('-')
                self.COLOR = color.from_str(color_str)

    def print(self):
        print('==== Config %s ====' % self.name)
        print('Net type: ', self.NET_TYPE)
        print('Encode dim: ', self.N_ENCODE_DIM)
        print('Normalize: ', self.NORMALIZE)
        print('Train with depth: ', self.DEPTH_REF)
        print('Support direction: ', False if self.N_DIR_ENCODE is None
              else f'encode to {self.N_DIR_ENCODE}')
        print('Full-connected network parameters:', self.FC_PARAMS)
        print('Sample parameters', self.SAMPLE_PARAMS)
        if self.NERF_FINE_NET_PARAMS['enable']:
            print('NeRF fine network parameters', self.NERF_FINE_NET_PARAMS)
        print('==========================')

    def create_net(self):
        if self.NET_TYPE == 'msl':
            return MslNet(fc_params=self.FC_PARAMS,
                          sampler_params=self.SAMPLE_PARAMS,
                          normalize_coord=self.NORMALIZE,
                          c=self.COLOR,
                          encode_to_dim=self.N_ENCODE_DIM)
        if self.NET_TYPE == 'mslray':
            return MslRay(fc_params=self.FC_PARAMS,
                          sampler_params=self.SAMPLE_PARAMS,
                          normalize_coord=self.NORMALIZE,
                          c=self.COLOR,
                          encode_to_dim=self.N_ENCODE_DIM)
        if self.NET_TYPE == 'mslfast':
            return MslFast(fc_params=self.FC_PARAMS,
                           sampler_params=self.SAMPLE_PARAMS,
                           normalize_coord=self.NORMALIZE,
                           c=self.COLOR,
                           encode_to_dim=self.N_ENCODE_DIM)
        if self.NET_TYPE == 'msl2fast':
            return MslFast(fc_params=self.FC_PARAMS,
                           sampler_params=self.SAMPLE_PARAMS,
                           normalize_coord=self.NORMALIZE,
                           c=self.COLOR,
                           encode_to_dim=self.N_ENCODE_DIM,
                           include_r=True)
        if self.NET_TYPE == 'nerf':
            return CascadeNerf(fc_params=self.FC_PARAMS,
                               sampler_params=self.SAMPLE_PARAMS,
                               fine_params=self.NERF_FINE_NET_PARAMS,
                               normalize_coord=self.NORMALIZE,
                               c=self.COLOR,
                               coord_encode=self.N_ENCODE_DIM,
                               dir_encode=self.N_DIR_ENCODE)
        if self.NET_TYPE == 'nerf2':
            return CascadeNerf2(fc_params=self.FC_PARAMS,
                                sampler_params=self.SAMPLE_PARAMS,
                                normalize_coord=self.NORMALIZE,
                                c=self.COLOR,
                                coord_encode=self.N_ENCODE_DIM,
                                dir_encode=self.N_DIR_ENCODE)
        if self.NET_TYPE == 'nerfbg':
            return CascadeNerf(fc_params=self.FC_PARAMS,
                               sampler_params=self.SAMPLE_PARAMS,
                               fine_params=self.NERF_FINE_NET_PARAMS,
                               normalize_coord=self.NORMALIZE,
                               c=self.COLOR,
                               coord_encode=self.N_ENCODE_DIM,
                               dir_encode=self.N_DIR_ENCODE,
                               bg_layer=True)
        if self.NET_TYPE == 'bgnet':
            return BgNet(fc_params=self.FC_PARAMS,
                         encode=self.N_ENCODE_DIM,
                         c=self.COLOR)
        if self.NET_TYPE.startswith('oracle'):
            return Oracle(fc_params=self.FC_PARAMS,
                          sampler_params=self.SAMPLE_PARAMS,
                          normalize_coord=self.NORMALIZE,
                          coord_encode=self.N_ENCODE_DIM,
                          out_activation=self.NET_TYPE[6:] if len(self.NET_TYPE) > 6 else 'sigmoid')
        if self.NET_TYPE.startswith('cnerf'):
            return CNerf(fc_params=self.FC_PARAMS,
                         sampler_params=self.SAMPLE_PARAMS,
                         c=self.COLOR,
                         coord_encode=self.N_ENCODE_DIM,
                         n_bins=int(self.NET_TYPE[5:] if len(self.NET_TYPE) > 5 else 128))
        if self.NET_TYPE.startswith('dnerfa'):
            return NerfDepth(fc_params=self.FC_PARAMS,
                             sampler_params=self.SAMPLE_PARAMS,
                             c=self.COLOR,
                             coord_encode=self.N_ENCODE_DIM,
                             n_bins=int(self.NET_TYPE[7:] if len(self.NET_TYPE) > 7 else 128),
                             include_neighbor_bins=False)
        if self.NET_TYPE.startswith('dnerf'):
            return NerfDepth(fc_params=self.FC_PARAMS,
                             sampler_params=self.SAMPLE_PARAMS,
                             c=self.COLOR,
                             coord_encode=self.N_ENCODE_DIM,
                             n_bins=int(self.NET_TYPE[6:] if len(self.NET_TYPE) > 6 else 128))
        if self.NET_TYPE.startswith('nnerf'):
            return NNerf(fc_params=self.FC_PARAMS,
                         sampler_params=self.SAMPLE_PARAMS,
                         n_nets=int(self.NET_TYPE[5:] if len(self.NET_TYPE) > 5 else 1),
                         normalize_coord=self.NORMALIZE,
                         c=self.COLOR,
                         coord_encode=self.N_ENCODE_DIM,
                         dir_encode=self.N_DIR_ENCODE)
        if self.NET_TYPE.startswith('snerffastx'):
            return SnerfFast(fc_params=self.FC_PARAMS,
                             sampler_params=self.SAMPLE_PARAMS,
                             n_parts=int(self.NET_TYPE[10:] if len(self.NET_TYPE) > 10 else 1),
                             normalize_coord=self.NORMALIZE,
                             c=self.COLOR,
                             coord_encode=self.N_ENCODE_DIM,
                             dir_encode=self.N_DIR_ENCODE,
                             multiple_net=False)
        if self.NET_TYPE.startswith('snerffast'):
            return SnerfFast(fc_params=self.FC_PARAMS,
                             sampler_params=self.SAMPLE_PARAMS,
                             n_parts=int(self.NET_TYPE[9:] if len(self.NET_TYPE) > 9 else 1),
                             normalize_coord=self.NORMALIZE,
                             c=self.COLOR,
                             coord_encode=self.N_ENCODE_DIM,
                             dir_encode=self.N_DIR_ENCODE)
        if self.NET_TYPE.startswith('nmsl'):
            n_nets = int(self.NET_TYPE[4:]) if len(self.NET_TYPE) > 4 else 2
            if self.SAMPLE_PARAMS['n_samples'] % n_nets != 0:
                raise ValueError('n_samples should be divisible by n_nets')
            return NewMslNet(fc_params=self.FC_PARAMS,
                             sampler_params=self.SAMPLE_PARAMS,
                             normalize_coord=self.NORMALIZE,
                             n_nets=n_nets,
                             c=self.COLOR,
                             encode_to_dim=self.N_ENCODE_DIM)
        raise ValueError('Invalid net type')
