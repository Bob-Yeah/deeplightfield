def update_config(config):
    # Net parameters
    config.NET_TYPE = 'snerffastx4'
    config.N_ENCODE_DIM = 6
    #config.N_DIR_ENCODE = 4
    config.FC_PARAMS.update({
        'nf': 512,
        'n_layers': 8
    })
    config.SAMPLE_PARAMS.update({
        'depth_range': (0.3, 7),
        'n_samples': 128,
        'perturb_sample': False
    })
