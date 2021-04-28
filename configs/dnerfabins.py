def update_config(config):
    # Net parameters
    config.NET_TYPE = 'dnerfa'
    config.N_ENCODE_DIM = 10
    config.DEPTH_REF = True
    config.FC_PARAMS.update({
        'nf': 256,
        'n_layers': 8
    })
    config.SAMPLE_PARAMS.update({
        'depth_range': (0.4, 6),
        'n_samples': 8,
        'perturb_sample': False
    })