import os
import numpy as np

import pdb

from mathutils import Matrix,Vector
import torch

import kornia
# out_dir = './database/'


from tqdm import tqdm

ori_dir = '/projects/perception/datasets/animal_videos/version9/'
animal_list = os.listdir('./database')



for animal in animal_list:
    rtk_out_dir = os.path.join('./database',animal,'RTK')

    os.makedirs(rtk_out_dir,exist_ok=True)


    cam_out_dir = os.path.join('./database',animal,'Camera')
    os.makedirs(cam_out_dir,exist_ok=True)

    for frame in tqdm(range(180)):
        info_dir = os.path.join(ori_dir,animal,'info','{:04d}.npz'.format(frame+1))
        info = np.load(info_dir)

        ## xyz to opengl
        rot_mat = info['cam_rot'][[1,2,0],:]

        loc_mat = info['cam_loc'][[1,2,0]]

        depth = np.linalg.norm(loc_mat)


        intrin = info['intrinsic_mat']
        extrin = info['extrinsic_mat']

        quat = np.array(kornia.rotation_matrix_to_quaternion(torch.tensor(rot_mat)))

        result_cam = np.zeros(8)
        result_cam[0] = intrin[0,0]/512

        result_cam[1] = 0
        result_cam[2] = 0

        result_cam[3] = quat[3]
        result_cam[4:7] = quat[:3]

        result_cam[7] = depth.item()

##################################################

        result_rtk = np.eye(4)
        result_rtk[:3,:3] = rot_mat
        result_rtk[:3,-1] = loc_mat
        result_rtk[-1] = np.array([intrin[0,0],intrin[0,0],intrin[0,-1],intrin[1,-1]])
        # result[:3] = location
        # result[3:] = quat

        np.savetxt(os.path.join(rtk_out_dir,'{:05d}.txt'.format(frame+1)),result_rtk)
        np.savetxt(os.path.join(cam_out_dir,'{:05d}.txt'.format(frame+1)),result_cam)

        # pdb.set_trace()

