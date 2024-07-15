

def load_my_args(opt):
    """Manually set the arguments that would be loaded from the terminal command. Useful for running finetune in the GUI."""
    
    #('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    opt.batch_size = 64
    # ('--epochs', default=50, type=int)
    opt.epochs = 5
    
    # Model parameters
    # ('--input_size', default=224, type=int, help='images input size')
    opt.input_size = 224
    
    # Optimizer parameters
    # ('--lr', type=float, default=1e-4, metavar='LR',help='learning rate (absolute lr)')
    opt.lr = 1e-4
    # ('--warmup_epoch_percentage', type=int, default=.1, metavar='N', help='epochs to warmup LR')
    opt.warmup_epoch_percentage = 0.1
    # ('--weight_decay', type=float, default=1e-4, help='weight decay (default: 1e-4)') 1e-4 = 0.0001
    opt.weight_decay = 1e-4
    
    # Dataset parameters
    # ('--data_path', default='./IDRiD_data', type=str, help='dataset path')
    opt.data_path = './data/IDRiD_data/'
    # ('--nb_classes', default=5, type=int, help='number of the classification types')
    opt.nb_classes = 5
    # ('--output_dir', default='./output_dir/', help='path where to save, empty for no saving')
    opt.output_dir = './output_dir/'
    # ('--resume', default='', help='resume from checkpoint')
    opt.resume = ''
    # ('--eval', action='store_true', help='Perform evaluation only')
    opt.eval = True
    
    # * Cutmix params
    # ('--cutmix', type=float, default=0., help='cutmix alpha, cutmix enabled if > 0.')
    opt.cutmix = 0.0
    
    # * Finetuning params
    # ('--finetune', default='./RETFound_oct_weights.h5',type=str, help='finetune from checkpoint') #name of the pretrained model to continue using
    opt.finetune = ''
    # ('--task', default='',type=str, help='finetune from checkpoint')
    opt.task = ''
    # ('--global_pool', action='store_true')
    opt.global_pool = True
    # (global_pool=True)
    # ('--cls_token', action='store_false', dest='global_pool', help='Use class token instead of global pool for classification')
    opt.cls_token = False

    return opt