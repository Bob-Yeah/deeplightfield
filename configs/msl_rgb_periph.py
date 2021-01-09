def update_config(config):
    # Dataset settings
    config.GRAY = False

    # Net parameters
    config.NET_TYPE = 'msl'
    config.N_ENCODE_DIM = 10
    config.FC_PARAMS.update({
        'nf': 64,
        'n_layers': 8
    })
    config.SAMPLE_PARAMS.update({
        'depth_range': (1, 50),
        'n_samples': 4
    })