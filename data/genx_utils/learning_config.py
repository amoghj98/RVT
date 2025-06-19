import argparse
import os
import torch
from datetime import datetime
import json
import numpy as np
import logging
import sys
import cv2

def print_args(args):
    #logging.info args
    logging.info(' ' * 20 + 'OPTIONS:')
    for k, v in vars(args).items():
        logging.info(' ' * 20 + k + ': ' + str(v)) 

def configs():
    # model_names = sorted(name for name in models.__dict__
    #                  if name.islower() and not name.startswith("__"))
    parser = argparse.ArgumentParser(description='MOTE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #FILE HANDLING args
    parser.add_argument('--root-dir', 
                        type=str, 
                        metavar='DIR', 
                        default='/local/a/datasets/',
                        help='path to dataset')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='MulObjTrack_Dataset', 
                        choices=['MulObjTrack_Dataset', 
                                 'simulatedDataset', 
                                 'MVSEC'], 
                        help="Path on which generated dataset is to be saved") 
    parser.add_argument('--sub-dir', 
                        type=str, 
                        metavar='DIR', 
                        default='processed_v2',
                        choices=['processed', 
                                 'processed_v2'],
                        help='path to dataset')
    parser.add_argument('--groupname', 
                        type=str, 
                        default='objects', 
                        choices=['dummy', 'path', 'objects'], 
                        help="The group from which data is to be visualised")
    parser.add_argument('--single-speed', 
                        type=int, 
                        default=None, 
                        help='use data from a single speed bin hdf5 file')
    
    parser.add_argument('--override-path', type=str, default=None, help='Override Data path')
    
    parser.add_argument("--descriptor", type=str, default=None,
                        help="Run descriptor")         
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed')
    parser.add_argument("--save-path",
                        help="Save results path",
                        default='./results')
    
    #TRAINING args
    parser.add_argument('--method', 
                        metavar='MODEL', 
                        choices=['tuning_and_clustering'
                                'tuning_and_learning',
                                'fully_learning'],
                        help='Method',
                        default='fully_learning')
    
    parser.add_argument('--model-type', 
                        choices=['dotie_snn',
                                'ofs_snn', 'ofs_snn_2l',
                                'ofp_ann', 'ofp_snn',
                                'ofp_ann_lks', 'ofp_snn_lks',
                                'ofd_ann', 'ofd_snn',
                                'ofd_ann_lks', 'ofd_snn_lks',
                                'ofpd_ann', 'ofpd_snn'
                                'ofpd_ann_lks', 'ofpd_snn_lks',
                                'ofpd_snn_seq', 'ofpd_snn_lks_seq',
                                'ofs_ofpd' #Purely Evaluation Model
                                ],
                        help='Model type to use, overwritten if pretrained is specified',
                        default='ofd_ann')
    
    parser.add_argument('--solver', 
                        choices=['adam','sgd'],
                        help='solver algorithms',
                        default='adam')
    parser.add_argument('--loss', 
                        choices=['MSE','L1'],
                        help='Loss Criterion',
                        default='L1')
    parser.add_argument('--ofs-loss', 
                        choices=['MSE','L1'],
                        help='Loss Criterion',
                        default='MSE')
    parser.add_argument('-j', '--num-workers', 
                        type=int, 
                        help='number of data loading workers',
                        default=8)
    parser.add_argument('--epochs', 
                        type=int, 
                        help='number of total epochs to run',
                        default=60)
    parser.add_argument('--start_epoch', 
                        type=int, 
                        help='starting epoch',
                        default=0)
    parser.add_argument('-b', '--batch-size', 
                        type=int, 
                        help='mini-batch size',
                        default=512)
    parser.add_argument('-t', '--test-batch-size', 
                        type=int,
                        help='mini-batch size',
                        default=512)
    parser.add_argument('--lr', '--learning-rate', 
                        type=float,
                        metavar='LR', 
                        help='initial learning rate',
                        default=0.005)
    parser.add_argument('--lrd', '--learning-rate-decay',
                        type=float,
                        help='Rate at which the learning rate is decayed.',
                        default=0.8)
    parser.add_argument('--momentum', 
                        type=float, 
                        metavar='M',
                        help='momentum for sgd, alpha parameter for adam',
                        default=0.9)
    parser.add_argument('--beta', 
                        type=float, 
                        help='beta parameter for adam',
                        default=0.999)
    
    #Norm Layer Params
    parser.add_argument('--norm-layer',
                        choices=['batchnorm','layernorm', 'instancenorm', 'bntt', 'lntt', 'intt', 'none'],
                        help='Normalization type',
                        default='instancenorm')
    parser.add_argument('--affine',
                        type=bool, 
                        help='affine parameter of norm layer',
                        default=True)
    parser.add_argument('--trs',
                        type=bool, 
                        help='track_running_stats parameter of norm layer',
                        default=False)
    
    parser.add_argument('--ann-milestones', 
                        metavar='N', 
                        nargs='*', 
                        help='epochs at which learning rate is multiplied by decay factor',
                        default= [10,20,30,40,50,60,70,80,90,100])#[20,40,60,80,120,160])
    parser.add_argument('--snn-milestones', 
                        metavar='N', 
                        nargs='*', 
                        help='epochs at which learning rate is multiplied by decay factor',
                        # default=[5,10,15,20,25,30,40,50,60,80,120,160])
                        default= [10,20,30,40,50,60,70,80,90,100])
    parser.add_argument('--half', 
                        action='store_true', 
                        help='Use Half precision',
                        default=False)
    parser.add_argument('--base-channels', 
                        type=int, 
                        help='base channels for the model',
                        default=8)



    #DATASET args
    parser.add_argument('--dt', 
                        type=int, 
                        default=1000,
                        choices=[100, 200, 500, 1000, 2000, 5000, 10000], 
                        help='Input representation time interval in us')
    parser.add_argument('--nBins',
                        type=int,
                        help='Number of equi-event bins (timesteps) within dt',
                        default=5)
    parser.add_argument('--nSpeeds',
                        type=int,
                        help='Number of speeds',
                        default=4)
    parser.add_argument('--min-speed',
                        type=int,
                        help='Min Speed',
                        default=1)
    parser.add_argument('--width',
                        type=int,
                        help='Width of event frame',
                        default=640)
    parser.add_argument('--height',
                        type=int,
                        help='Width of event frame',
                        default=480)
    # parser.add_argument('--speed_range_means', 
    #                     type=float, 
    #                     default=[6.0, 12.0, 24.0, 36.0, 48.0, 74.0, 96.0, 144.0], 
    #                     help='Mean values of velocity in m/s for each velocity bin')
    # parser.add_argument('--speed_range_mins', 
    #                     type=float,  
    #                     default=[4.0, 8.0,  18.0, 30.0, 42.0, 64.0, 84.0, 120.0], 
    #                     help='Minimum values of velocity in m/s for each velocity bin')
    # parser.add_argument('--speed_range_maxs', 
    #                     type=float,  
    #                     default=[8.0, 18.0, 30.0, 42.0, 64.0, 84.0, 120.0, 500.0], 
    #                     help='Maximum values of velocity in m/s for each velocity bin')
    parser.add_argument('--speed_range_means', 
                        type=float, 
                        default=[9.0, 30.0, 63.0, 120.0], 
                        help='Mean values of velocity in m/s for each velocity bin')
    parser.add_argument('--speed_range_mins', 
                        type=float,  
                        default=[1.0, 18.0, 42.0, 84.0], 
                        help='Minimum values of velocity in m/s for each velocity bin')
    parser.add_argument('--speed_range_maxs', 
                        type=float,  
                        default=[18.0, 42.0, 84.0, 500.0], 
                        help='Maximum values of velocity in m/s for each velocity bin')

    parser.add_argument('--inp-downscale',
                        type=int,
                        help='Scale down input by a factor',
                        default=2)
    parser.add_argument('--data-reduction',
                        type=int,
                        help='Reduce data by a factor',
                        default=1)
    parser.add_argument('--normalize',
                        type=bool, 
                        help='normalize groundtruth to (0,1) while training',
                        default=False)
    
    #NOISE args
    parser.add_argument('--ofpd-noise',
                        type=float, 
                        help='Fraction of pixels to add noise to, no noise if 0',
                        default=0)
    parser.add_argument('--ofs-noise',
                        type=float, 
                        help='Fraction of pixels to add noise to, no noise if 0',
                        default=0)
    
    


    # TESTING args
    parser.add_argument('--pretrained',
                        type=str,
                        help="Pretrained model path")
    parser.add_argument('--render',
                        action='store_true',
                        help='If true, the predictions will be visualized during testing.',
                        default=False)
    parser.add_argument('--save-vis',
                        action='store_true',
                        help='If true, the predictions will be stored as GIFs.',
                        default=False)
    parser.add_argument('--save-mulObj-vis',
                        action='store_true',
                        help='If true, the predictions will be stored as GIFs.',
                        default=False)
    parser.add_argument('--mulObj-n', 
                        type=float, 
                        help='Multi-Object Sequence to Evaluate',
                        default=1)
    parser.add_argument('--evaluate-at', nargs='*', metavar='E',
                        help='Evaluate at these epochs ',
                        default=[1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,120,140,160,180,200])
    parser.add_argument('--visualize-at', nargs='*', metavar='E',
                        help='Visualize at these epochs ',
                        default=[1,10,20,40,60,80,100,120,160,200])
    parser.add_argument('-e', '--evaluate', 
                        dest='evaluate', 
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--no-ts',
                        action='store_true',
                        help='If false append timestamp to save-path',
                        default=False)
    parser.add_argument('--gpus', 
                        type=str, 
                        help='gpus (default: 0)',
                        default='0')
    parser.add_argument('--debug',
                        action='store_true',
                        help='If true, debugging renders enabled for DOTIE+DBSCAN Tuning',
                        default=False)
    parser.add_argument('--pTH',
                        type=int, 
                        help='Threshold for number of non-zero pixels',
                        default=[0, 8, 45, 150, 120]) #250
                        # default=[0, 120, 180, 260, 320]) #250
    

    #SNN args
    parser.add_argument('--ofsthresh', 
                        type=float, 
                        help='Initial Threshold (default: 1.0)',
                        default=0.2)
    parser.add_argument('--ofsleak', 
                        type=float,  
                        help='Initial Leak = No leak (default: 1.0)',
                        default=1.0)
    parser.add_argument('--learn-thresh', 
                        default=False,
                        action='store_true',
                        help='Learnable Threshold')
    parser.add_argument('--learn-leak', 
                        default=False,
                        action='store_true',
                        help='Learnable Leak')
    parser.add_argument('--reset-mechanism', 
                        type=str, 
                        choices=['subtract', 'hard', 'noreset'], 
                        help='Reset mechanism',
                        default='subtract')
    parser.add_argument('--ofs-speed', 
                        type=int, 
                        help='Speed to train the model for',
                        default=4)
    parser.add_argument('--ofs-mode', 
                        type=str, 
                        help='OFS learning mode',
                        choices=['fixed_speed', 'min_speed'],
                        default='min_speed')
    parser.add_argument('--spk-neuron', 
                        type=str, 
                        choices=['lif', 'snntorch_lif'],
                        help='Speed to train the model for',
                        default='lif')
    parser.add_argument('--sigmoid-leak', 
                        default=False,
                        action='store_true',
                        help='Apply Sigmoid on Leak')

    #REALTIME-EVAL ARGS
    parser.add_argument('--arrow-len', 
                        default=40, 
                        type=int, 
                        help='Directional Arrow Len')

    args=parser.parse_args() 

    args.codec=cv2.VideoWriter_fourcc(*'mp4v')

    if args.dataset == 'MulObjTrack_Dataset':     
        args.traindir = os.path.join(args.root_dir, args.dataset, args.sub_dir, str(args.dt)+'us', 'train')
        args.testdir = os.path.join(args.root_dir, args.dataset, args.sub_dir, str(args.dt)+'us', 'test')
        args.visdir = os.path.join(args.root_dir, args.dataset, args.sub_dir, str(args.dt)+'us', 'vis')
        args.mulObjdir = os.path.join(args.root_dir, args.dataset, args.sub_dir, str(args.dt)+'us', 'mulObj')
        args.groupname = 'objects'

    elif args.dataset == 'MulObjTrack_Dataset_old': 
        if args.prefix == 'Sequences_20us':
            args.hdf5_path = os.path.join(args.root_dir, args.dataset, 'Sequences_20us', args.seq_name + '.hdf5')
            args.groupname = 'dummy'
        elif args.prefix == 'dummy_sequences':
            args.hdf5_path = os.path.join(args.root_dir, args.dataset, 'dummy_sequences', args.seq_name + '.hdf5') 
            args.groupname = 'dummy'
        else:
            print("hdf5_test.py: Error: Dataset File not found. Is arg hdf5Filename correct?")
            sys.exit(1)

    elif args.dataset == 'simulatedDataset':
            args.hdf5_path = os.path.join(args.root_dir, args.dataset, 'baylands_day1.hdf5')
            args.groupname = 'path'
    
    elif args.dataset == 'MVSEC':
            args.testfile = os.path.join(args.root_dir, args.dataset, 'indoor_flying1', 'indoor_flying1_data.hdf5')
            args.groupname = 'path'

    os.environ['CUDA_VISIBLE_DEVICES']= args.gpus
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%m-%d_%H-%M")

    #Thresholds and Leaks for 5-speed Tuning Method @ 2000us accumulation time for event-bins (args.render-time)
    # args.ithresh = [1.0, 5.5, 12.0, 18.0, 22.0]
    # args.ileak = [0.1, 0.05, 0.01, 0.04, 0.0001]

    #Thresholds and Leaks for Tuning Method @ 1000us accumulation time for event-bins (args.render-time)
    # args.ithresh = [1.0, 2.0, 6.2, 9.0, 12.0, 16.0, 22.0, 27.0]
    # args.ileak = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    #Thresholds and Leaks for Tuning Method @ 1000us accumulation time for PROCESSED event-bins (args.render-time)
    # args.ithresh = [1.0, 2.0, 6.2, 9.0, 12.0, 16.0, 22.0, 25.0]
    # args.ileak = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    #Thresholds and Leaks for four_speed_tracking_data.py  dataset=processed_v2
    if args.dt == 1000:
        args.ithresh = [0.5, 1.8, 2.2, 3.1]
        args.ileak = [0.1, 0.1, 0.1, 0.2]
    elif args.dt == 500:
        args.ithresh = [0.2, 1.4, 2.0, 2.6]
        args.ileak = [0.1, 0.1, 0.1, 0.2]
    elif args.dt == 200:
        args.ithresh = [0.1, 0.9, 1.4, 2.1]
        args.ileak = [0.1, 0.1, 0.1, 0.2]
    else:
        args.ithresh = [0.5, 1.8, 2.2, 3.1]
        args.ileak = [0.1, 0.1, 0.1, 0.2]


    assert (args.nSpeeds + args.min_speed) <= 9

    if args.evaluate:
        if 'snn' in args.model_type:
            args.learn_mode = 'snn_learnable'
            args.milestones = args.snn_milestones
        elif 'ann' in args.model_type:
             args.learn_mode='ann'   
             args.milestones = args.ann_milestones

    if args.learn_thresh and args.learn_leak:
        args.learn_mode = 'snn_learnable'
        args.milestones = args.snn_milestones
    elif args.learn_thresh:
        args.learn_mode = 'snn_thresh'
        args.milestones = args.snn_milestones
    elif args.learn_leak:
        args.learn_mode = 'snn_leak'
        args.milestones = args.snn_milestones
    else:
        if 'snn' in args.model_type:
            args.learn_mode = 'snn'
            args.milestones = args.snn_milestones
        elif 'ann' in args.model_type:
             args.learn_mode='ann'   
             args.milestones = args.ann_milestones

    if 'ntt' in args.norm_layer:
        args.model_type = args.model_type + '_ntt'

    
    args.data_reduction_train = {10000: [11, 3.7, 1.5, 1],      #4100 bins/speed, data_reduction=1
                                 5000: [9.5, 3.2, 1.4, 1],      #10000 bins/speed, data_reduction=1
                                 2000: [12, 4.4, 2, 1.3],       #10000 bins/speed, data_reduction=2
                                 1000: [10, 4, 1.9, 1.2],       #10000 bins/speed, data_reduction=5
                                 500: [10, 4, 2.2, 1.3],        #20000 bins/speed, data_reduction=5
                                 200: [18, 6.8, 4.5, 2.3],      #30000 bins/speed, data_reduction=5
                                 100: [26, 11.7, 7.2, 3.7]}     #40000 bins/speed, data_reduction=5
    
    args.data_reduction_test =  {10000: [20, 4.8, 1.3, 2.6],      #1000 bins/speed, data_reduction=1
                                 5000: [15, 4, 1.5, 2.5],       #2000 bins/speed, data_reduction=1
                                 2000: [18, 5, 2.5, 3.5],       #2000 bins/speed, data_reduction=2    
                                 1000: [8, 3, 1, 1.5],          #2000 bins/speed, data_reduction=5
                                 500: [10, 3, 1.5, 2],          #3500 bins/speed, data_reduction=5
                                 200: [18, 8, 4, 4],            #4500 bins/speed, data_reduction=5
                                 100: [21, 10.2, 5.6, 4.8]}     #8000 bins/speed, data_reduction=5


    red = (0, 0, 255) #speed8
    magenta = (255, 0, 255) #speed7
    violet = (255, 0, 128) #speed6
    orange = (0, 165, 255) #speed5
    yellow = (0, 255, 255) #speed4
    cyan = (255, 255, 0) #speed3
    green = (0, 255, 0) #speed2
    lgreen = (144, 238, 144) #speed1

    # args.colors = [lgreen, green, cyan, yellow, orange, violet, magenta, red] #8 Speeds
    args.colors = [green, yellow, orange, red] #4 Speeds

    args.white = (255, 255, 255)
    args.gray = (120, 120, 120)

    if 'ofs_ofpd' in args.model_type:
        if args.no_ts:
            if args.descriptor is not None:
                args.save_path = os.path.join(args.save_path, args.model_type, args.descriptor, 'n'+str(int(args.ofs_noise*100)), "dt"+str(args.dt))
            else:
                args.save_path = os.path.join(args.save_path, args.model_type, 'n'+str(int(args.ofs_noise*100)), "dt"+str(args.dt))
        else:
            if args.descriptor is not None:
                args.save_path = os.path.join(args.save_path, args.model_type, args.descriptor, 'n'+str(int(args.ofs_noise*100)), "dt"+str(args.dt), timestamp)
            else:
                args.save_path = os.path.join(args.save_path, args.model_type, 'n'+str(int(args.ofs_noise*100)), "dt"+str(args.dt), timestamp)
    elif 'ofs' in args.model_type:
        args.arch = 'speed' + str(args.ofs_speed)
        if args.no_ts:
            if args.descriptor is not None:
                args.save_path = os.path.join(args.save_path, args.model_type, args.descriptor, args.arch, 'n'+str(int(args.ofs_noise*100)), "dt"+str(args.dt))
            else:
                args.save_path = os.path.join(args.save_path, args.model_type, args.arch, 'n'+str(int(args.ofs_noise*100)), "dt"+str(args.dt))
        else:
            if args.descriptor is not None:
                args.save_path = os.path.join(args.save_path, args.model_type, args.descriptor, args.arch, 'n'+str(int(args.ofs_noise*100)), "dt"+str(args.dt), timestamp)
            else:
                args.save_path = os.path.join(args.save_path, args.model_type, args.arch, 'n'+str(int(args.ofs_noise*100)), "dt"+str(args.dt), timestamp)

    else: #ofp, ofd, ofpd
        args.arch = 'bc' + str(args.base_channels)
        if args.descriptor is not None:
            args.save_path = os.path.join(args.save_path, args.model_type, args.descriptor, "dt"+str(args.dt), timestamp)
        else:
            args.save_path = os.path.join(args.save_path, args.model_type, "dt"+str(args.dt), timestamp)   
        
    if 'ofs_snn_2l' in args.model_type:
        args.ofs_mode = 'fixed_speed'

    if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    if args.save_vis:
         args.vis_path = os.path.join(args.save_path, 'vis')
         if not os.path.exists(args.vis_path):
            os.makedirs(args.vis_path)

    if args.save_mulObj_vis:
         args.mulObj_vis_path = os.path.join(args.save_path, 'mulObj')
         if not os.path.exists(args.mulObj_vis_path):
            os.makedirs(args.mulObj_vis_path)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.save_path, "logfile.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

    #Seed random number generators
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Dump config
    with open(os.path.join(args.save_path, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return args

    