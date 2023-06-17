SEED = 42
unet_settings = {'opt': 'adamw',
                 'lr': {'sgd': 0.01, 'adam': 0.001, 'adamw': 0.001},
                 'decay': 0.0001, 'grad_clip': -1,
                 }
segtran_settings = {'opt': 'adamw',
                    'lr': {'adamw': 0.0002},
                    'decay': 0.0001, 'grad_clip': 0.1,
                    'dropout_prob': {'234': 0.3, '34': 0.2, '4': 0.2},
                    'num_modes': {'234': 2, '34': 4, '4': 4}
                    }

default_settings = {
    'unet': unet_settings,
    'unet-scratch': unet_settings,
    'nestedunet': unet_settings,
    'unet3plus': unet_settings,
    'deeplabv3plus': unet_settings,
    'deeplab-smp': unet_settings,
    'pranet': unet_settings,
    'attunet': unet_settings,
    'r2attunet': unet_settings,
    'dunet': unet_settings,
    'nnunet': unet_settings,
    'setr': segtran_settings,
    'transunet': segtran_settings,
    'segtran': segtran_settings,
    'fundus': {
        'num_classes': 3,
        'bce_weight': [0., 1, 2],
        'ds_class': 'SegCrop',
        'ds_names': 'train,valid,test,drishti,rim',
        'orig_input_size': 576,
        # Each dim of the patch_size should be multiply of 32.
        'patch_size': 288,
        'uncropped_size': {
            'train': (2056, 2124),
            'test': (1634, 1634),
            'valid': (1634, 1634),
            'valid2': (1940, 1940),
            'test2': -1,  # varying sizes
            'drishti': (2050, 1750),
            'rim': (2144, 1424),
            'train-cyclegan': (2056, 2124),
            'rim-cyclegan': (2144, 1424),
            'gamma-train': -1,  # varying sizes
            'gamma-valid': -1,  # varying sizes
            'gamma-test': -1,  # varying sizes
        },
        'has_mask': {
            'train': True, 'test': True,
            'valid': True, 'valid2': False,
            'test2': False,
            'drishti': True, 'rim': True,
            'train-cyclegan': True,
            'rim-cyclegan': True,
            'gamma-train': True,
            'gamma-valid': False,
            'gamma-test': False},
        'weight': {
            'train': 1, 'test': 1,
            'valid': 1, 'valid2': 1,
            'test2': 1,
            'drishti': 1, 'rim': 1,
            'train-cyclegan': 1,
            'rim-cyclegan': 1,
            'gamma-train': 1,
            'gamma-valid': 1,
            'gamma-test': 1
        },
        # if the uncropped_size of a dataset == -1, then its orig_dir
        # has to be specified here for the script to acquire
        # the uncropped_size of each image.
        'orig_dir': {
            'test2': 'test2_orig',
            'gamma-train': 'gamma_train_orig/images',
            'gamma-valid': 'gamma_valid_orig/images',
            'gamma-test': 'gamma_test_orig/images'
        },
        'orig_ext': {
            'test2': '.jpg',
            'gamma-train': '.png',
            'gamma-valid': '.jpg',
            'gamma-test': '.jpg'
        },
    },
    'polyp': {
        'num_classes': 2,
        'bce_weight': [0., 1],
        'ds_class': 'SegWhole',
        'ds_names': 'CVC-ClinicDB-train,Kvasir-train',
        # Actual images are at various sizes. As the dataset is SegWhole, orig_input_size is ignored.
        # But output_upscale is computed as the ratio between orig_input_size and patch_size.
        # So set it to the same as patch_size to avoid output upscaling.
        # Setting orig_input_size to -1 also leads to output_upscale = 1.
        # All images of different sizes are resized to 320*320.
        'orig_input_size': 320,
        'patch_size': 320,
        'has_mask': {'CVC-ClinicDB-train': True, 'Kvasir-train': True,
                     'CVC-ClinicDB-test': True, 'Kvasir-test': True,
                     'CVC-300': True,
                     'CVC-ClinicDB-train-cyclegan': True,
                     'CVC-300-cyclegan': True,
                     'CVC-ColonDB': False,
                     'ETIS-LaribPolypDB': True},
        'weight': {'CVC-ClinicDB-train': 1, 'Kvasir-train': 1,
                   'CVC-ClinicDB-test': 1, 'Kvasir-test': 1,
                   'CVC-300': 1,
                   'CVC-ClinicDB-train-cyclegan': 1,
                   'CVC-300-cyclegan': 1,
                   'CVC-ColonDB': 1,
                   'ETIS-LaribPolypDB': 1}
    },
    'tcich': {
        'num_classes': 3,
        'bce_weight': [0., 1],
        'ds_class': 'SegWhole',
        'ds_names': 'train',
        # Actual images are at various sizes. As the dataset is SegWhole, orig_input_size is ignored.
        # But output_upscale is computed as the ratio between orig_input_size and patch_size.
        # So set it to the same as patch_size to avoid output upscaling.
        # Setting orig_input_size to -1 also leads to output_upscale = 1.
        # All images of different sizes are resized to 320*320.
        'orig_input_size': 320,
        'patch_size': 320,
        'has_mask': {'train': True, 'test': True},
        'weight': {'train': 1, 'test': 1}
    },
    'oct': {
        'num_classes': 10,
        'bce_weight': [0., 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'ds_class': 'SegWhole',
        'ds_names': 'duke',
        # Actual images are at various sizes. As the dataset is SegWhole, orig_input_size is ignored.
        # But output_upscale is computed as the ratio between orig_input_size and patch_size.
        # If you want to avoid avoid output upscaling, set orig_input_size to the same as patch_size.
        # The actual resolution of duke is (296, 500~542).
        # Set to (288, 512) will crop the central areas.
        # The actual resolution of pcv is (633, 720). Removing 9 pixels doesn't matter.
        'orig_input_size': {'duke': (288, 512), 'seed': (1024, 512), 'pcv': (624, 720)},
        'patch_size': {'duke': (288, 512), 'seed': (512, 256), 'pcv': (312, 360)},
        'has_mask': {'duke': True, 'seed': False, 'pcv': False},
        'weight': {'duke': 1, 'seed': 1, 'pcv': 1}
    },
}
