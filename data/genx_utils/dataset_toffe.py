import torch
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import logging
import sys
import cv2
import pdb

from data.utils.types import LoaderDataDictGenX, DataType

from functools import partialmethod
from pathlib import Path
from typing import List, Union

from omegaconf import DictConfig
from torchdata.datapipes.map import MapDataPipe
from tqdm import tqdm

from data.genx_utils.sequence_for_streaming import SequenceForIter, RandAugmentIterDataPipe
from data.genx_utils.labels import ObjectLabels, SparselyBatchedObjectLabels
from data.utils.stream_concat_datapipe import ConcatStreamingDataPipe
from data.utils.stream_sharded_datapipe import ShardedStreamingDataPipe
from data.utils.types import DatasetMode, DatasetType

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils.visual_helpers import normalize, convert_to_color

class MOTE_Dataset_Parallel(Dataset):
    def __init__(self, config, mode='train'):
        self.root_dir = "/scratch/gautschi/joshi157/datasets/processed_v2/"
        self.dt = 1000
        self.ofpd_noise = 0
        self.traindir = os.path.join(self.root_dir, str(self.dt)+'us', 'train')
        self.testdir = os.path.join(self.root_dir, str(self.dt)+'us', 'test')
        self.visdir = os.path.join(self.root_dir, str(self.dt)+'us', 'vis')
        self.mulObjdir = os.path.join(self.root_dir, str(self.dt)+'us', 'mulObj')
        self.groupname = 'objects'
        self.data_reduction_train = {10000: [11, 3.7, 1.5, 1],      #4100 bins/speed, data_reduction=1
                                 5000: [9.5, 3.2, 1.4, 1],      #10000 bins/speed, data_reduction=1
                                 2000: [12, 4.4, 2, 1.3],       #10000 bins/speed, data_reduction=2
                                 1000: [10, 4, 1.9, 1.2],       #10000 bins/speed, data_reduction=5
                                 500: [10, 4, 2.2, 1.3],        #20000 bins/speed, data_reduction=5
                                 200: [18, 6.8, 4.5, 2.3],      #30000 bins/speed, data_reduction=5
                                 100: [26, 11.7, 7.2, 3.7]}     #40000 bins/speed, data_reduction=5     
        self.data_reduction_test =  {10000: [20, 4.8, 1.3, 2.6],      #1000 bins/speed, data_reduction=1
                                5000: [15, 4, 1.5, 2.5],       #2000 bins/speed, data_reduction=1
                                2000: [18, 5, 2.5, 3.5],       #2000 bins/speed, data_reduction=2    
                                1000: [8, 3, 1, 1.5],          #2000 bins/speed, data_reduction=5
                                500: [10, 3, 1.5, 2],          #3500 bins/speed, data_reduction=5
                                200: [18, 8, 4, 4],            #4500 bins/speed, data_reduction=5
                                100: [21, 10.2, 5.6, 4.8]}     #8000 bins/speed, data_reduction=5

        # self.args = args
        self.dt = 1000  # default from learning_config.py
        self.nBins = 10  # default from learning_config.py
        self.nSpeeds = 4  # default from learning_config.py
        self.groupname = 'objects'  # default from learning_config.py
        self.scaling_factor = 2  # default inp_downscale from learning_config.py
        # self.data_reduction = args.data_reduction

        self.height = 480  # default from learning_config.py
        self.width = 640  # default from learning_config.py

        if mode == DatasetMode.TRAIN:
            self.datadir = self.traindir
            self.data_reduction = self.data_reduction_train
            logging.info(f'\nTraining DataSet...')
        elif mode == DatasetMode.TESTING:
            self.datadir = self.visdir
            self.data_reduction = 1
            logging.info(f'\nTesting DataSet...')
        elif mode == DatasetMode.VALIDATION:
            self.datadir = self.testdir
            self.data_reduction = self.data_reduction_test
            logging.info(f'\nValidation DataSet...')

        self.sf = {}
        self.sg = {}
        self.length = 0
        self.speedbin_lens = []
        self.speedbin_event_lens = []
        self.invalid = 0
        for sidx in range(self.nSpeeds):
            self.sf[sidx] = h5py.File(os.path.join(self.datadir, 'speed_bin' + str(sidx+1) +'.hdf5'), "r")
            self.sg[sidx] = self.sf[sidx][self.groupname]
            self.speedbin_lens.append(int(self.sg[sidx]['pos_txed'].shape[0]//self.data_reduction[self.dt][sidx]))
            corr_index = self.sg[sidx]['speed_corr_map'][self.speedbin_lens[sidx]].item()
            # pdb.set_trace()
            self.speedbin_event_lens.append(self.sg[sidx]['events'][:corr_index].shape[0])
            self.length += self.speedbin_lens[sidx]

        self.speedbin_cumsum = np.cumsum(np.array(self.speedbin_lens))
        
        logging.info(f'Dataset Keys: {self.sg[0].keys()}')
        logging.info(f'Speed Bin Lengths: {self.speedbin_lens}')
        logging.info(f'Speed Bin Cumsum: {self.speedbin_cumsum}')
        logging.info(f'Speed Bin Event Lengths: {self.speedbin_event_lens}')
        logging.info(f'Total Dataset Length: {self.length}')

        self.events = {}
        self.pos_gt = {}
        self.speed_gt = {}
        self.dir_gt = {}
        self.corr_map = {}

        logging.info(f'Reading HDF5 data into RAM! Please be patient...')

        for idx in range(self.nSpeeds):
            self.events[idx] = self.sg[idx]['events'][:self.speedbin_event_lens[idx]]
            self.pos_gt[idx] = self.sg[idx]['pos_txed'][:self.speedbin_lens[idx]]
            self.speed_gt[idx] = self.sg[idx]['mod_vel_mps'][:self.speedbin_lens[idx]]
            self.dir_gt[idx] = self.sg[idx]['vel_hding'][:self.speedbin_lens[idx]]
            self.corr_map[idx] = self.sg[idx]['speed_corr_map'][:self.speedbin_lens[idx]+1]

        for sidx in range(self.nSpeeds):
            self.sf[sidx].close()
        
        logging.info('[DONE]')
    
    def addNoise(self, eventBins, noise_level=0.1):
        # eventBins shape: (480, 640, nBins) / self.scaling_factor
        nBins = eventBins.shape[2]
        noisy_events = eventBins.copy()
        numNoisyPix = np.ceil(noise_level*eventBins.shape[0]*eventBins.shape[1])
        y = np.random.randint(0, eventBins.shape[0]-1, int(numNoisyPix))[:, np.newaxis]
        x = np.random.randint(0, eventBins.shape[1]-1, int(numNoisyPix))[:, np.newaxis]
        for bin in range(nBins):
            noisy_events[y, x, bin] = eventBins[:, :, bin].max()
        return noisy_events

    def get_speed_bin(self, index):
        split = index//self.speedbin_cumsum

        if split.any() == False: #all zeros => 1st speed bin
            hdf5_idx = 0
            inp_idx = index
        else:
            hdf5_idx = split.nonzero()[0][-1] + 1 #last (highest) non-zero element - The last element of split will NEVER be 1 as index is always lower than it
            inp_idx = index - self.speedbin_cumsum[hdf5_idx-1]

        return hdf5_idx, inp_idx

    def get_event_bins(self, events, valid_pose=True):
        # event_bins = np.zeros((self.height//self.scaling_factor, self.width//self.scaling_factor, self.nBins), dtype=float) #Currently merging polarities
        event_bins = np.zeros((self.height//self.scaling_factor, self.width//self.scaling_factor, 2 * self.nBins), dtype=float) #Unmerged polarities
        
        if valid_pose:
            total_events = events.shape[0]
            # logging.info(events.shape)

            # for bin in range(self.nBins):
            #     bin_start = int(bin*total_events//self.nBins)
            #     bin_end = int((bin+1)*total_events//self.nBins)
            #     evx = (events[bin_start:bin_end, 0]//self.scaling_factor).astype(int)
            #     evy = (events[bin_start:bin_end, 1]//self.scaling_factor).astype(int)
            #     event_bins[evy, evx, bin] += 1
            #     if bin == 0:
            #         logging.info(evx.shape)
            #         logging.info(evy.shape)
            #         logging.info(np.unique(events[:, 3]))
                    #
            for bin in range(self.nBins):
                bin_start = int(bin*total_events//self.nBins)
                bin_end = int((bin+1)*total_events//self.nBins)
                evx = (events[bin_start:bin_end, 0]//self.scaling_factor).astype(int)
                evy = (events[bin_start:bin_end, 1]//self.scaling_factor).astype(int)
                pol = (events[bin_start:bin_end, 3]).astype(int)
                # +ve events in even bins
                event_bins[evy, evx, (2 * bin) + pol] += 1
        else:
            self.invalid += 1

        return event_bins
    
    def get_valid_pose(self, pos_txed):
        xlimit = 25
        ylimit = 25

        valid_pose = True

        if ((pos_txed[0]-xlimit) < 0) or ((pos_txed[0]+xlimit) > self.width//self.scaling_factor):
            valid_pose = False

        if ((pos_txed[1]-ylimit) < 0) or ((pos_txed[1]+ylimit) > self.height//self.scaling_factor):
            valid_pose = False

        return valid_pose


    def __getitem__(self, index):
        hdf5_idx, inp_idx = self.get_speed_bin(index)

        gt_pos_txed = self.pos_gt[hdf5_idx][inp_idx]//self.scaling_factor
        gt_pos_txed[0] = np.clip(gt_pos_txed[0], 0, self.width//self.scaling_factor-1)
        gt_pos_txed[1] = np.clip(gt_pos_txed[1], 0, self.height//self.scaling_factor-1)

        gt_mod_vel_mps = self.speed_gt[hdf5_idx][inp_idx]
        gt_vel_hding = self.dir_gt[hdf5_idx][inp_idx]
        gt_vel_hding_degrees = 180*self.dir_gt[hdf5_idx][inp_idx]/np.pi
        gt_vel_hding_degrees[gt_vel_hding_degrees<0] += 360.0

        gt_dir_x = np.cos(gt_vel_hding)
        gt_dir_y = -np.sin(gt_vel_hding)
        gt_dir = np.concatenate((gt_dir_x, gt_dir_y), axis=0)[:, np.newaxis]

        evf = self.corr_map[hdf5_idx][inp_idx]
        evl = self.corr_map[hdf5_idx][inp_idx+1]
        events = self.events[hdf5_idx][int(evf):int(evl)]

        valid_pose = self.get_valid_pose(gt_pos_txed)

        event_bins = self.get_event_bins(events, valid_pose)
        if self.ofpd_noise !=0:
            event_bins = self.addNoise(event_bins, noise_level=self.ofpd_noise)

        ev_repr = torch.permute(torch.from_numpy(event_bins), (2, 0, 1))
        ev_repr = ev_repr.unsqueeze(0)
        # logging.info(ev_repr.shape) # [5, 240, 320]
        # logging.info(gt_dir.shape) # [2, 1]
        # logging.info(gt_pos_txed.shape) # [2, 1]
        if valid_pose:
            # obj_labels_seq = torch.cat((torch.from_numpy(gt_pos_txed), torch.from_numpy(gt_dir)), dim=0)
            # Create a tensor with the required format: [t, x, y, w, h, class_id, class_confidence]
            label_tensor = torch.zeros((1, 7), dtype=torch.float32)
            label_tensor[0, 0] = 0  # t
            label_tensor[0, 1] = torch.from_numpy(gt_pos_txed[0]//self.scaling_factor).item()  # x
            label_tensor[0, 2] = torch.from_numpy(gt_pos_txed[1]//self.scaling_factor).item()  # y
            label_tensor[0, 3] = 50  # w (fixed width)
            label_tensor[0, 4] = 50  # h (fixed height)
            label_tensor[0, 5] = 0  # class_id (assuming single class)
            label_tensor[0, 6] = 1.0  # class_confidence
            # label_tensor = label_tensor.unsqueeze(0)
            obj_labels = ObjectLabels(label_tensor, input_size_hw=(self.height//self.scaling_factor, self.width//self.scaling_factor))
        else:
            # obj_labels_seq = torch.cat((torch.from_numpy(gt_pos_txed), torch.from_numpy(np.zeros_like(torch.from_numpy(gt_dir)))), dim=0)
            # obj_labels = ObjectLabels.create_empty()
            obj_labels = None

        # Create SparselyBatchedObjectLabels with a single frame
        sparse_obj_labels = SparselyBatchedObjectLabels([obj_labels])

        # Add debug logging
        # logging.info(f"Event representation shape: {ev_repr.shape}")
        # logging.info(f"Valid pose: {valid_pose}")
        # logging.info(f"Label tensor shape: {label_tensor.shape if valid_pose else 'None'}")
        # logging.info(f"Object labels: {obj_labels}")

        return {
            DataType.EV_REPR: ev_repr,
            DataType.OBJLABELS_SEQ: sparse_obj_labels,
            DataType.IS_FIRST_SAMPLE: False,
            DataType.IS_PADDED_MASK: False,
        }
        # if valid_pose:
        #     return torch.from_numpy(event_bins), torch.from_numpy(gt_pos_txed), torch.from_numpy(gt_dir), torch.from_numpy(gt_mod_vel_mps), torch.from_numpy(gt_vel_hding_degrees)#, self.invalid
        # else:
        #     return torch.from_numpy(event_bins), torch.from_numpy(np.zeros_like(gt_pos_txed)), torch.from_numpy(np.zeros_like(gt_dir)), torch.from_numpy(np.zeros_like(gt_mod_vel_mps)), torch.from_numpy(np.zeros_like(gt_vel_hding_degrees))

    def __len__(self):
        return self.length

# def visualize_ofs_data(args, event_inp, event_gt, speed_gt):
#     eventFrame = event_inp[0].sum(2).numpy()
#     eventFrame= normalize(eventFrame)
#     event_color_frame = convert_to_color(eventFrame)

#     eventGT = event_gt[0].sum(2).numpy()
#     eventGT= normalize(eventGT)
#     event_color_gt = convert_to_color(eventGT)

#     cv2.putText(event_color_frame, 'Speed (m/s) : ' + str(int(speed_gt[0].numpy())), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
#     return event_color_frame, event_color_gt

# def visualize_data(args, event_inp, pos_gt, dir_gt, vel_mag, vel_hding):
#     eventFrame = event_inp[0].sum(2).numpy()
#     eventFrame= normalize(eventFrame)
#     event_color_frame = convert_to_color(eventFrame)

#     # pdb.set_trace()
#     event_diff_frame = (event_inp[0,:,:,-1] - event_inp[0,:,:,0]).numpy()
#     event_diff_frame[abs(event_diff_frame) > 0] = 1
#     event_diff_frame = normalize(event_diff_frame)
#     event_dcolor_frame = convert_to_color(event_diff_frame)

#     pos_txed_gt = pos_gt[0].numpy()
#     dir_gt = dir_gt[0].numpy()
#     vel_mps_gt = vel_mag[0].numpy()
#     vel_hding_d_gt = vel_hding[0].numpy()
#     vel_hding = vel_hding*np.pi/180

#     # pt_gt = np.array([np.cos(vel_hding), -np.sin(vel_hding)])
#     pt_gt = dir_gt

#     # pdb.set_trace()

#     l=1
#     centerx =  int(pos_txed_gt[0].item())
#     centery =  int(pos_txed_gt[1].item())

#     eptx = centerx+int(l*vel_mps_gt*pt_gt[0].item())
#     epty = centery+int(l*vel_mps_gt*pt_gt[1].item())

#     cv2.circle(event_color_frame, (centerx, centery) , 8//args.inp_downscale, (255, 0, 0), 5//args.inp_downscale)
#     cv2.arrowedLine(event_color_frame, (centerx, centery), (eptx, epty), (0, 0, 255), 2)
#     cv2.putText(event_color_frame, str(int(vel_hding_d_gt)), (centerx, centery+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
#     cv2.putText(event_color_frame, 'Speed (m/s) : ' + str(int(vel_mps_gt)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

#     return event_color_frame, event_dcolor_frame

def build_toffe_dataset(dataset_mode: DatasetMode, dataset_config: DictConfig) -> MOTE_Dataset_Parallel:
    return MOTE_Dataset_Parallel(mode=dataset_mode, config=dataset_config)

if __name__ == '__main__':
    args = configs()
    print_args(args)
    
    TrainSet = MOTE_Dataset_Parallel(args, mode='test')
    TrainLoader = torch.utils.data.DataLoader(dataset=TrainSet, batch_size=1, shuffle=False)

    # VisSet = VIS_Dataset(args, mode='vis')
    # VisLoader = torch.utils.data.DataLoader(dataset=VisSet, batch_size=1, shuffle=False)

    # OFSSet = OFS_Dataset(args, mode='test', speed=args.ofs_speed)
    # OFSLoader = torch.utils.data.DataLoader(dataset=OFSSet, batch_size=1, shuffle=False)

    input(...)

    #MOTE and VIS  Test
    for idx, input in enumerate(TrainLoader):
        event_inp, pos_gt, dir_gt, vel_mag, vel_hding = input
        # pdb.set_trace()
        # if len(torch.nonzero(event_inp[0].sum(2))) > 0:
        print(f'Idx: {idx},    nEvents: {len(event_inp.nonzero())},    Pose: [{int(pos_gt[0].squeeze(1)[0])}, {int(pos_gt[0].squeeze(1)[1])}],    Direction: {(vel_hding[0].numpy()).round(5).item()},    Speed: {vel_mag[0].numpy().round(5).item()}', end='\r', flush=True)
        vis_frame, dcolor_frame = visualize_data(args, event_inp, pos_gt, dir_gt, vel_mag, vel_hding)

        # vis_frame = cv2.resize(vis_frame,(720,480))
        cv2.imshow('Event Data with GT', vis_frame)
        # cv2.imshow('Event Bin Diff', dcolor_frame)
        cv2.waitKey(1)
    

    #OFS Test
    # for idx, input in enumerate(OFSLoader):
    #     event_inp, event_gt, speed_gt = input
    #     # pdb.set_trace()
    #     print(f'Idx: {idx},   nEvents: {len(event_inp.nonzero())},  Speed: {speed_gt[0].numpy().round(5).item()}', end='\r', flush=True)
    #     event_frame, gt_frame = visualize_ofs_data(args, event_inp, event_gt, speed_gt)

    #     # vis_frame = cv2.resize(vis_frame,(720,480))
    #     cv2.imshow('Event Frame', event_frame)
    #     cv2.imshow('GT Frame', gt_frame)
    #     # cv2.imshow('Event Bin Diff', dcolor_frame)
    #     cv2.waitKey(1)

#Useful Command Line Arguments
#--inp-downscale : Factor to scale down the input size from  [480, 640] - keep at 2
#--data-reduction: Factor to scale down the dataset (train/test) - Higher value 10-100 for visualizing all speeds
#-j : Number of workers to load data - keep at 1
#Pay attention to dataset chracteristics printed in terminal for lengths of individual speed_binX.hdf5 loaded with data reduction