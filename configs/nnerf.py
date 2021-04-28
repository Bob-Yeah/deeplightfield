def update_config(config):
    # Net parameters
    config.NET_TYPE = 'nnerf4'
    config.N_ENCODE_DIM = 10
    config.FC_PARAMS.update({
        'nf': 128,
        'n_layers': 4
    })
    config.SAMPLE_PARAMS.update({
        'depth_range': (1, 50),
        'n_samples': 32,
        'perturb_sample': True
    })
