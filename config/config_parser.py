import argparse


def configure_parse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ###### General arguments ######
    parser.add_argument('--task', dest='task_name', type=str, default='fundus', help='Name of the segmentation task.')
    parser.add_argument('--ds', dest='ds_names', type=str, default=None,
                        help='Dataset folders. Can specify multiple datasets (separated by ",")')
    parser.add_argument('--split', dest='ds_split', type=str, default='all',
                        help='Split of the dataset (Can specify the split individually for each dataset)')
    parser.add_argument("--profile", dest='do_profiling', action='store_true', default=False,
                        help='Calculate amount of params and FLOPs. ')

    parser.add_argument('--insize', dest='orig_input_size', type=str, default=None,
                        help='Use images of this size (among all cropping sizes) for training. Set to 0 to use all sizes.')
    parser.add_argument('--patch', dest='patch_size', type=str, default=None,
                        help='Resize input images to this size for training.')

    ###### Few-shot learning arguments ######
    parser.add_argument('--samplenum', dest='sample_num', type=str, default=None,
                        help='Numbers of supervised training samples to use for each dataset (Default: None, use all images of each dataset. '
                             'Provide 0 for a dataset to use all images of it. Do not use -1 as it will cause errors of argparse).')
    parser.add_argument("--bnopt", dest='bn_opt_scheme', type=str, default=None,
                        choices=[None, 'fixstats', 'affine'],
                        help='How to optimize BN stats/affine params during training.')

    ###### Polyformer arguments ######
    parser.add_argument("--polyformer", dest='polyformer_mode', type=str, default=None,
                        choices=[None, 'none', 'source', 'target'],
                        help='Do polyformer traning.')
    parser.add_argument("--sourceopt", dest='poly_source_opt', type=str, default='allpoly',
                        help='What params to optimize on the source domain.')
    parser.add_argument("--targetopt", dest='poly_target_opt', type=str, default='k',
                        help='What params to optimize on the target domain.')
    ###### END of Polyformer arguments ######

    ###### Adversarial training arguments ######
    parser.add_argument("--adv", dest='adversarial_mode', type=str, default=None,
                        choices=[None, 'none', 'feat', 'mask'],
                        help='Mode of adversarial training.')
    parser.add_argument("--featdisinchan", dest='num_feat_dis_in_chan', type=int, default=64,
                        help='Number of input channels of the feature discriminator')

    parser.add_argument("--sourceds", dest='source_ds_names', type=str, default=None,
                        help='Dataset name of the source domain.')
    parser.add_argument("--sourcebs", dest='source_batch_size', type=int, default=-1,
                        help='Batch size of unsupervised adversarial learning on the source domain (access all source domain images).')
    parser.add_argument("--targetbs", dest='target_unsup_batch_size', type=int, default=-1,
                        help='Batch size of unsupervised adversarial learning on the target domain (access all target domain images).')
    parser.add_argument('--domweight', dest='DOMAIN_LOSS_W', type=float, default=0.002,
                        help='Weight of the adversarial domain loss.')
    parser.add_argument('--supweight', dest='SUPERVISED_W', type=float, default=1,
                        help='Weight of the supervised training loss. Set to 0 to do unsupervised DA.')
    parser.add_argument('--reconweight', dest='RECON_W', type=float, default=0,
                        help='Weight of the reconstruction loss for DA. Default: 0, no reconstruction.')
    parser.add_argument("--adda", dest='adda', action='store_true',
                        help='Use ADDA (instead of the default RevGrad objective).')

    ###### END of adversarial training arguments ######
    ###### END of few-shot learning arguments ######

    ###### Robustness experiment settings ######
    parser.add_argument("--optfilter", dest='opt_filters', type=str, default=None,
                        help='Only optimize params that match the specified keyword.')
    parser.add_argument("--robustaug", dest='robust_aug_types', type=str, default=None,
                        # Examples: None, 'brightness,contrast',
                        help='Augmentation types used during robustness training.')
    parser.add_argument("--robustaugdeg", dest='robust_aug_degrees', type=str, default='0.5,1.5',
                        help='Degrees of robustness augmentation (1 or 2 numbers).')
    parser.add_argument("--gbias", dest='use_global_bias', action='store_true',
                        help='Use the global bias instead of transformer layers.')

    ###### End of Robustness experiment settings ######

    parser.add_argument('--maxiter', type=int, default=10000, help='maximum training iterations')
    parser.add_argument('--saveiter', type=int, default=500, help='save model snapshot every N iterations')
    parser.add_argument('--cp', dest='checkpoint_path', type=str, default=None, help='Load this checkpoint')

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.25, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--channels', '-x', type=int, default=3,
                        help='Number of channels image')

    ###### Optimization settings ######
    parser.add_argument('--lrwarmup', dest='lr_warmup_steps', type=int, default=500, help='Number of LR warmup steps')
    parser.add_argument('--bs', dest='batch_size', type=int, default=4, help='Total batch_size on all GPUs')
    parser.add_argument('--opt', type=str, default=None, help='optimization algorithm')
    parser.add_argument('--lr', type=float, default=-1, help='learning rate')
    parser.add_argument('--decay', type=float, default=-1, help='weight decay')
    parser.add_argument('--gradclip', dest='grad_clip', type=float, default=-1, help='gradient clip')
    parser.add_argument('--attnclip', dest='attn_clip', type=int, default=500, help='Segtran attention clip')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--locprob", dest='localization_prob', default=0,
                        type=float, help='Probability of doing localization during training')
    parser.add_argument("--tunebn", dest='tune_bn_only', action='store_true',
                        help='Only tune batchnorms for domain adaptation, and keep model weights unchanged.')

    parser.add_argument('--diceweight', dest='MAX_DICE_W', type=float, default=0.5,
                        help='Weight of the dice loss.')
    parser.add_argument('--focus', dest='focus_class', type=int, default=-1,
                        help='The class that is particularly predicted (with higher loss weight)')
    parser.add_argument('--exclusive', dest='use_exclusive_masks', action='store_true',
                        help='Aim to predict exclulsive masks (instead of non-exclusive ones)')

    parser.add_argument("--vcdr", dest='vcdr_estim_scheme', type=str, default='none',
                        choices=['none', 'dual', 'single'],
                        help='The scheme of the learned vCDR loss for fundus images. none: not using vCDR loss. '
                             'dual:   estimate vCDR with an individual vC estimator and vD estimator. '
                             'single: estimate vCDR directly using a single CNN.')

    parser.add_argument("--vcdrweight", dest='VCDR_W', type=float, default=0.01,
                        help='Weight of vCDR loss.')
    parser.add_argument("--vcdrestimstart", dest='vcdr_estim_loss_start_iter', type=int, default=1000,
                        help='Start iteration of vCDR loss for the vCDR estimator.')
    # vCDR estimator usually converges very fast. So 100 iterations are enough.
    parser.add_argument("--vcdrnetstart", dest='vcdr_net_loss_start_iter', type=int, default=1100,
                        help='Start iteration of vCDR loss for the segmentation model.')

    ###### End of optimization settings ######

    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument("--debug", dest='debug', action='store_true', help='Debug program.')

    parser.add_argument('--net', type=str, default='segtran', help='Network architecture')
    parser.add_argument('--bb', dest='backbone_type', type=str, default='eff-b4',
                        help='Backbone of Segtran / Encoder of other models')
    parser.add_argument("--nopretrain", dest='use_pretrained', action='store_false',
                        help='Do not use pretrained weights.')

    ###### Transformer architecture settings ######
    parser.add_argument("--nosqueeze", dest='use_squeezed_transformer', action='store_false',
                        help='Do not use attractor transformers (Default: use to increase scalability).')
    parser.add_argument("--attractors", dest='num_attractors', default=256,
                        type=int, help='Number of attractors in the squeezed transformer.')
    parser.add_argument("--noqkbias", dest='qk_have_bias', action='store_false',
                        help='Do not use biases in Q, K projections (Using biases leads to better performance on BraTS).')

    parser.add_argument("--translayers", dest='num_translayers', default=1,
                        type=int, help='Number of Cross-Frame Fusion layers.')
    parser.add_argument('--layercompress', dest='translayer_compress_ratios', type=str, default=[1, 1],
                        help='Compression ratio of channel numbers of each transformer layer to save RAM.')
    parser.add_argument('--modes', type=int, dest='num_modes', default=1, help='Number of transformer modes')
    parser.add_argument('--multihead', dest='ablate_multihead', action='store_true',
                        help='Ablation to expanded transformer (using multihead instead)')

    parser.add_argument('--dropout', type=float, dest='dropout_prob', default=-1, help='Dropout probability')

    parser.add_argument('--pos', dest='pos_code_type', type=str, default='lsinu',
                        choices=['lsinu', 'none', 'rand', 'sinu', 'bias'],
                        help='Positional code scheme')
    parser.add_argument('--posw', dest='pos_code_weight', type=float, default=1.0)
    parser.add_argument('--posr', dest='pos_bias_radius', type=int, default=7,
                        help='The radius of positional biases')
    parser.add_argument("--squeezeuseffn", dest='has_FFN_in_squeeze', action='store_true',
                        help='Use the full FFN in the first transformer of the squeezed attention '
                             '(Default: only use the first linear layer, i.e., the V projection)')

    parser.add_argument("--attnconsist", dest='use_attn_consist_loss', action='store_true',
                        help='This loss encourages the attention scores to be consistent with the segmentation mask')
    parser.add_argument("--attnconsistweight", dest='ATTNCONSIST_W', type=float, default=0.01,
                        help='Weight of the attention consistency loss')

    ############## Mince transformer settings ##############
    parser.add_argument("--mince", dest='use_mince_transformer', action='store_true',
                        help='Use Mince (Multi-scale) Transformer to save GPU RAM.')
    parser.add_argument("--mincescales", dest='mince_scales', type=str, default=None,
                        help='A list of numbers indicating the mince scales.')
    parser.add_argument("--minceprops", dest='mince_channel_props', type=str, default=None,
                        help='A list of numbers indicating the relative proportions of channels of each scale.')
    ###### End of transformer architecture settings ######

    ###### Segtran (non-transformer part) settings ######
    parser.add_argument("--infpn", dest='in_fpn_layers', default='34',
                        choices=['234', '34', '4'],
                        help='Specs of input FPN layers')
    parser.add_argument("--outfpn", dest='out_fpn_layers', default='1234',
                        choices=['1234', '234', '34'],
                        help='Specs of output FPN layers')

    parser.add_argument("--outdrop", dest='out_fpn_do_dropout', action='store_true',
                        help='Do dropout on out fpn features.')
    parser.add_argument("--inbn", dest='in_fpn_use_bn', action='store_true',
                        help='Use BatchNorm instead of GroupNorm in input FPN.')
    parser.add_argument("--nofeatup", dest='bb_feat_upsize', action='store_false',
                        help='Do not upsize backbone feature maps by 2.')
    ###### End of Segtran (non-transformer part) settings ######

    ###### Augmentation settings ######
    # Using random scaling as augmentation usually hurts performance. Not sure why.
    parser.add_argument("--randscale", type=float, default=0.2, help='Do random scaling augmentation.')
    parser.add_argument("--affine", dest='do_affine', action='store_true', help='Do random affine augmentation.')
    parser.add_argument("--gray", dest='gray_alpha', type=float, default=0.5,
                        help='Convert images to grayscale by so much degree.')
    parser.add_argument("--reshape", dest='reshape_mask_type', type=str, default=None,
                        choices=[None, 'rectangle', 'ellipse'],
                        help='Intentionally reshape the mask to test how well the model fits the mask bias.')
    return parser
