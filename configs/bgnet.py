def update_config(config):
    # Net parameters
    config.NET_TYPE = 'bgnet'
    config.N_ENCODE_DIM = 10
    config.FC_PARAMS.update({
        'nf': 128,
        'n_layers': 4
    })
