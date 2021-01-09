import os
import importlib
from os.path import join


class SphericalViewSynConfig(object):

    def __init__(self):
        self.name = 'default'

        self.GRAY = False

        # Net parameters
        self.NET_TYPE = 'msl'
        self.N_ENCODE_DIM = 10
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
            'deeplightfield.' + module_name)
        config_module.update_config(self)
        self.name = module_name.split('.')[-1]

    def load_by_name(self, name):
        config_module = importlib.import_module(
            'deeplightfield.configs.' + name)
        config_module.update_config(self)
        self.name = name

    def to_id(self):
        net_type_id = "%s-%s" % (self.NET_TYPE, "gray" if self.GRAY else "rgb")
        encode_id = "_e%d" % self.N_ENCODE_DIM
        fc_id = "_fc%dx%d" % (self.FC_PARAMS['nf'], self.FC_PARAMS['n_layers'])
        skip_id = "_skip%s" % ','.join([
            '%d' % val
            for val in self.FC_PARAMS['skips']
        ]) if len(self.FC_PARAMS['skips']) > 0 else ""
        depth_id = "_d%d-%d" % (self.SAMPLE_PARAMS['depth_range'][0],
                               self.SAMPLE_PARAMS['depth_range'][1])
        samples_id = '_s%d' % self.SAMPLE_PARAMS['n_samples']
        neg_flags = '%s%s%s' % (
            'p' if not self.SAMPLE_PARAMS['perturb_sample'] else '',
            'l' if not self.SAMPLE_PARAMS['lindisp'] else '',
            'i' if not self.SAMPLE_PARAMS['inverse_r'] else ''
        )
        neg_flags = '_~' + neg_flags if neg_flags != '' else ''
        return "%s@%s%s%s%s%s%s%s" % (self.name, net_type_id, encode_id, fc_id, skip_id, depth_id, samples_id, neg_flags)

    def from_id(self, id: str):
        self.name, config_str = id.split('@')
        segs = config_str.split('_')
        for i, seg in enumerate(segs):
            if i == 0: # NetType
                self.NET_TYPE, color_mode = seg.split('-')
                self.GRAY = (color_mode == 'gray')
                continue
            if seg.startswith('e'): # Encode
                self.N_ENCODE_DIM = int(seg[1:])
                continue
            if seg.startswith('fc'): # Full-connected network parameters
                self.FC_PARAMS['nf'], self.FC_PARAMS['n_layers'] = (int(str) for str in seg[2:].split('x'))
                continue
            if seg.startswith('skip'): # Skip connection
                self.FC_PARAMS['skips'] = [int(str) for str in seg[4:].split(',')]
                continue
            if seg.startswith('d'): # Depth range
                self.SAMPLE_PARAMS['depth_range'] = tuple(float(str) for str in seg[1:].split('-'))
                continue
            if seg.startswith('s'): # Number of samples
                self.SAMPLE_PARAMS['n_samples'] = int(seg[1:])
                continue
            if seg.startswith('~'): # Negative flags
                self.SAMPLE_PARAMS['perturb_sample'] = (seg.find('p') < 0)
                self.SAMPLE_PARAMS['lindisp'] = (seg.find('l') < 0)
                self.SAMPLE_PARAMS['inverse_r'] = (seg.find('i') < 0)
                continue

    def print(self):
        print('==== Config %s ====' % self.name)
        print('Net type: ', self.NET_TYPE)
        print('Encode dim: ', self.N_ENCODE_DIM)
        print('Full-connected network parameters:', self.FC_PARAMS)
        print('Sample parameters', self.SAMPLE_PARAMS)
        print('==========================')
