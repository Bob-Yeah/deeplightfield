def update_config(config):
    # Dataset settings
    config.GRAY = True

    # Net parameters
    config.NET_TYPE = 'msl'
    config.N_ENCODE_DIM = 20
    config.FC_PARAMS.update({
        'nf': 64,
        'n_layers': 12,
    })
    config.SAMPLE_PARAMS.update({
        'depth_range': (1, 20),
        'n_samples': 16
    })