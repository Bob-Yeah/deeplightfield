def update_config(config):
    # Net parameters
    config.NET_TYPE = 'mslfast'
    config.N_ENCODE_DIM = 6
    config.FC_PARAMS.update({
        'nf': 256,
        'n_layers': 8
    })
    config.SAMPLE_PARAMS.update({
        'depth_range': (0.5, 5),
        'n_samples': 128,
        'perturb_sample': False
    })
