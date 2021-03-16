from my import color_mode

def update_config(config):
    # Dataset settings
    config.COLOR = color_mode.RGB

    # Net parameters
    config.NET_TYPE = 'msl'
    config.N_ENCODE_DIM = 10
    config.FC_PARAMS.update({
        'nf': 128,
        'n_layers': 4
    })
    config.SAMPLE_PARAMS.update({
        'depth_range': (1, 50),
        'n_samples': 8
    })