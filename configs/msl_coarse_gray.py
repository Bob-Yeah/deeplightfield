def update_config(config):
    # Dataset settings
    config.GRAY = True

    # Net parameters
    config.NET_TYPE = 'msl'
    config.N_ENCODE_DIM = 10
    config.FC_PARAMS = {
        'nf': 128,
        'n_layers': 8,
        'skips': [4]
    }
    config.SAMPLE_PARAMS = {
        'depth_range': (1, 50),
        'n_samples': 8,
        'perturb_sample': True
    }