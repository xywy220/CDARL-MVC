def get_default_config(data_name):
    if data_name in ['CUB']:
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
    elif data_name in ['BDGP']:
        return dict(
            latent=128,
            generators=dict(
                arch2=[1750, 1024, 1024, 1024, 256],
                arch1=[79, 1024, 1024, 1024, 256],
                arch3=[128, 1024, 1024, 1024, 256],
                batchnorm=True,
            ),
            discriminators=dict(
                arch1=[128, 64, 32, 5],
                arch2=[128, 64, 32, 5],
                arch4=[128, 64, 32, 5],
                arch3=[1829, 1024, 256, 128, 32],
            ),
            training=dict(
                dim_all=1829,
                cluster=5,
                alpha=0.7,
                seed=12,   
                batch_size=512,
                epoch=50,
                lr=1.0e-3,
            ),
            g_d_freq=1,
        )
    else:
        raise Exception('Undefined data_name')
