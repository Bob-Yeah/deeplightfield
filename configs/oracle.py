def update_config(config):
    # Net parameters
    config.NET_TYPE = 'oracle'
    config.N_ENCODE_DIM = 0
    config.FC_PARAMS.update({
        'nf': 256,
        'n_layers': 8,
        'activation': 'selu',
    })
    config.SAMPLE_PARAMS.update({
        'depth_range': (0.4, 6),
        'n_samples': 128,
        'perturb_sample': False
    })
