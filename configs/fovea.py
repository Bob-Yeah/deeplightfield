def update_config(config):
    # Net parameters
    config.NET_TYPE = 'snerffast4'
    config.N_ENCODE_DIM = 10
    #config.N_DIR_ENCODE = 4
    config.FC_PARAMS.update({
        'nf': 256,
        'n_layers': 8
    })
    config.SAMPLE_PARAMS.update({
        'depth_range': (0.7, 10),
        'n_samples': 64,
        'perturb_sample': False
    })
