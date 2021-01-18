from ..my import color_mode

def update_config(config):
    # Dataset settings
    config.COLOR = color_mode.RGB

    # Net parameters
    config.NET_TYPE = 'nmsl'
    config.N_ENCODE_DIM = 10
    config.FC_PARAMS.update({
        'nf': 64,
        'n_layers': 4
    })
    config.SAMPLE_PARAMS.update({
        'depth_range': (1, 50),
        'n_samples': 16
    })