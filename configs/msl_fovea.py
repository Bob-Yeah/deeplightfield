def update_config(config):
    # Net parameters
    config.NET_TYPE = 'msl'
    config.FC_PARAMS.update({
        'nf': 128,
        'n_layers': 4
    })
    config.SAMPLE_PARAMS.update({
        'depth_range': (1, 50),
        'n_samples': 32
    })