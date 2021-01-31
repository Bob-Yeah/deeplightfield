import os
import importlib
from os.path import join
from ..my import color_mode
from ..nets.msl_net import MslNet
from ..nets.msl_net_new import NewMslNet
from ..nets.spher_net import SpherNet


class SphericalViewSynConfig(object):

    def __init__(self):
        self.name = 'default'

        self.COLOR = color_mode.RGB

        # Net parameters
        self.NET_TYPE = 'msl'
        self.N_ENCODE_DIM = 10
        self.NORMALIZE = False
        self.DIR_AS_INPUT = False
        self.OPT_DECAY = 0
        self.FC_PARAMS = {
            'nf': 256,
            'n_layers': 8,
            'skips': []
        }
        self.SAMPLE_PARAMS = {
            'spherical': True,
            'depth_range': (1, 50),
            'n_samples': 32,
            'perturb_sample': True,
            'lindisp': True,
            'inverse_r': True,
        }

    def load(self, path):
        module_name = os.path.splitext(path)[0].replace('/', '.')
        config_module = importlib.import_module(
            'deep_view_syn.' + module_name)
        config_module.update_config(self)
        self.name = module_name.split('.')[-1]

    def load_by_name(self, name):
        config_module = importlib.import_module(
            'deep_view_syn.configs.' + name)
        config_module.update_config(self)
        self.name = name

    def to_id(self):
        net_type_id = "%s-%s" % (self.NET_TYPE, color_mode.to_str(self.COLOR))
        encode_id = "_e%d" % self.N_ENCODE_DIM
        fc_id = "_fc%dx%d" % (self.FC_PARAMS['nf'], self.FC_PARAMS['n_layers'])
        skip_id = "_skip%s" % ','.join([
            '%d' % val
            for val in self.FC_PARAMS['skips']
        ]) if len(self.FC_PARAMS['skips']) > 0 else ""
        depth_id = "_d%.2f-%.2f" % (self.SAMPLE_PARAMS['depth_range'][0],
                                    self.SAMPLE_PARAMS['depth_range'][1])
        samples_id = '_s%d' % self.SAMPLE_PARAMS['n_samples']
        opt_decay_id = '_decay%.1e' % self.OPT_DECAY if self.OPT_DECAY > 1e-5 else ''
        neg_flags = '%s%s%s' % (
            'p' if not self.SAMPLE_PARAMS['perturb_sample'] else '',
            'l' if not self.SAMPLE_PARAMS['lindisp'] else '',
            'i' if not self.SAMPLE_PARAMS['inverse_r'] else ''
        )
        neg_flags = '_~' + neg_flags if neg_flags != '' else ''
        pos_flags = '%s%s' % (
            'n' if self.NORMALIZE else '',
            'd' if self.DIR_AS_INPUT else '',
        )
        pos_flags = '_+' + pos_flags if pos_flags != '' else ''
        return "%s@%s%s%s%s%s%s%s%s%s" % (self.name, net_type_id, encode_id, fc_id, skip_id, depth_id, samples_id, opt_decay_id, neg_flags, pos_flags)

    def from_id(self, id: str):
        id_splited = id.split('@')
        if len(id_splited) == 2:
            self.name = id_splited[0]
        segs = id_splited[-1].split('_')
        for i, seg in enumerate(segs):
            if seg.startswith('fc'):  # Full-connected network parameters
                self.FC_PARAMS['nf'], self.FC_PARAMS['n_layers'] = (
                    int(str) for str in seg[2:].split('x'))
                continue
            if seg.startswith('skip'):  # Skip connection
                self.FC_PARAMS['skips'] = [int(str)
                                           for str in seg[4:].split(',')]
                continue
            if seg.startswith('decay'):
                self.OPT_DECAY = float(seg[5:])
                continue
            if seg.startswith('e'):  # Encode
                self.N_ENCODE_DIM = int(seg[1:])
                continue
            if seg.startswith('d'):  # Depth range
                self.SAMPLE_PARAMS['depth_range'] = tuple(
                    float(str) for str in seg[1:].split('-'))
                continue
            if seg.startswith('s'):  # Number of samples
                self.SAMPLE_PARAMS['n_samples'] = int(seg[1:])
                continue
            if seg.startswith('~'):  # Negative flags
                if seg.find('p') >= 0:
                    self.SAMPLE_PARAMS['perturb_sample'] = False
                if seg.find('l') >= 0:
                    self.SAMPLE_PARAMS['lindisp'] = False
                if seg.find('i') >= 0:
                    self.SAMPLE_PARAMS['inverse_r'] = False
                continue
            if seg.startswith('+'):  # Positive flags
                if seg.find('n') >= 0:
                    self.NORMALIZE = True
                if seg.find('d') >= 0:
                    self.DIR_AS_INPUT = True
                continue
            if i == 0:  # NetType
                self.NET_TYPE, color_str = seg.split('-')
                self.COLOR = color_mode.from_str(color_str)
                continue

    def print(self):
        print('==== Config %s ====' % self.name)
        print('Net type: ', self.NET_TYPE)
        print('Encode dim: ', self.N_ENCODE_DIM)
        print('Optimizer decay: ', self.OPT_DECAY)
        print('Normalize: ', self.NORMALIZE)
        print('Direction as input: ', self.DIR_AS_INPUT)
        print('Full-connected network parameters:', self.FC_PARAMS)
        print('Sample parameters', self.SAMPLE_PARAMS)
        print('==========================')

    def create_net(self):
        if self.NET_TYPE == 'msl':
            return MslNet(fc_params=self.FC_PARAMS,
                          sampler_params=self.SAMPLE_PARAMS,
                          normalize_coord=self.NORMALIZE,
                          dir_as_input=self.DIR_AS_INPUT,
                          color=self.COLOR,
                          encode_to_dim=self.N_ENCODE_DIM)
        if self.NET_TYPE.startswith('nmsl'):
            n_nets = int(self.NET_TYPE[4:]) if len(self.NET_TYPE) > 4 else 2
            if self.SAMPLE_PARAMS['n_samples'] % n_nets != 0:
                raise ValueError('n_samples should be divisible by n_nets')
            return NewMslNet(fc_params=self.FC_PARAMS,
                             sampler_params=self.SAMPLE_PARAMS,
                             normalize_coord=self.NORMALIZE,
                             n_nets=n_nets,
                             dir_as_input=self.DIR_AS_INPUT,
                             color=self.COLOR,
                             encode_to_dim=self.N_ENCODE_DIM)
        if self.NET_TYPE == 'nnmsl':
            return NewMslNet(fc_params=self.FC_PARAMS,
                             sampler_params=self.SAMPLE_PARAMS,
                             normalize_coord=self.NORMALIZE,
                             dir_as_input=self.DIR_AS_INPUT,
                             not_same_net=True,
                             color=self.COLOR,
                             encode_to_dim=self.N_ENCODE_DIM)
        raise ValueError('Invalid net type')