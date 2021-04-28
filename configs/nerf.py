def update_config(config):
    # Net parameters
    config.NET_TYPE = 'nerf'
    config.N_ENCODE_DIM = 10
    config.FC_PARAMS.update({
        'nf': 256,
        'n_layers': 8
    })
    config.SAMPLE_PARAMS.update({
        'depth_range': (1, 50),
        'n_samples': 64,
        'perturb_sample': True
    })
    config.NERF_FINE_NET_PARAMS.update({
        'enable': True,
        'nf': 256,
        'n_layers': 8,
        'additional_samples': 128
    })