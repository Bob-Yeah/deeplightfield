def update_config(config):
    # Dataset settings
    config.GRAY = False

    # Net parameters
    config.NET_TYPE = 'msl'
    config.N_ENCODE_DIM = 10
    config.FC_PARAMS = {
        'nf': 64,
        'n_layers': 12,
        'skips': []
    }
    config.SAMPLE_PARAMS = {
        'depth_range': (1, 20),
        'n_samples': 16,
        'perturb_sample': True
    }