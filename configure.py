def get_default_config(data_name):
    if data_name in ['Reuters_dim10']:
        return dict(
            latent=16,
            generators=dict(
                arch2=[10, 1024, 1024, 1024, 256],
                arch1=[10, 1024, 1024, 1024, 256],
                arch3=[16, 1024, 1024, 1024, 256],
                batchnorm=True,
            ),
            discriminators=dict(
                arch1=[16, 10, 5],
                arch2=[16, 10, 5],
                arch4=[16, 10, 5],
                arch3=[20, 10, 5],
            ),
            training=dict(
                dim_all=20,
                cluster=6,
                alpha=1,
                seed=14,
                batch_size=1024,
                epoch=80,
                lr=1.0e-3,
            ),
            g_d_freq=1,
        )
    elif data_name in ['CUB']:
        return dict(
            latent=32,
            generators=dict(
                arch2=[300, 1024, 1024, 1024, 256],
                arch1=[1024, 1024, 1024, 1024, 256],
                arch3=[32, 1024, 1024, 1024, 256],
                batchnorm=True,
            ),
            discriminators=dict(
                arch1=[32, 32, 5],
                arch2=[32, 32, 5],
                arch4=[32, 32, 5],
                arch3=[1324, 1024, 256, 128, 32],
            ),
            training=dict(
                dim_all=1324,
                cluster=10,
                alpha=0.7,
                seed=8,
                batch_size=256,
                epoch=50,
                lr=1.0e-3,
            ),
            g_d_freq=1,
        )
    else:
        raise Exception('Undefined data_name')
