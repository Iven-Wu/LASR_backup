# Copyright 2021 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.insert(0,'third_party')

import numpy as np
import trimesh
import torch
import cv2
import kornia
import pdb

import ext_utils.flowlib as flowlib
import ext_utils.util_flow as util_flow
from ext_utils.io import mkdir_p
import soft_renderer as sr
import argparse
from PIL import Image
parser = argparse.ArgumentParser(description='render data')
parser.add_argument('--outdir', default='syn-spot3f',
                    help='output dir')
parser.add_argument('--model', default='spot',
                    help='model to render, {spot, eagle}')
parser.add_argument('--nframes', default=3,type=int,
                    help='number of frames to render')
parser.add_argument('--alpha', default=1.,type=float,
                    help='0-1, percentage of a full cycle')
args = parser.parse_args()


def read_obj(obj_path, for_open_mesh=False):
    with open(obj_path) as file:
        flag = 0
        points = []
        normals = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == 'o' and flag == 0:
                flag = 1
                continue
            elif strs[0] == 'o' and flag == 1:
                break
            if strs[0] == 'v':
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))

            if strs[0] == 'vn':
                normals.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == 'f':
                single_line_face = strs[1:]

                f_co = []
                for sf in single_line_face:
                    face_tmp = sf.split('/')[0]
                    f_co.append(face_tmp)
                if for_open_mesh == False:
                    if len(f_co) == 3:
                        faces.append((int(f_co[0]), int(f_co[1]), int(f_co[2])))
                    elif len(f_co) == 4:
                        faces.append((int(f_co[0]), int(f_co[1]), int(f_co[2])))
                        faces.append((int(f_co[0]), int(f_co[1]), int(f_co[3])))
                        faces.append((int(f_co[1]), int(f_co[2]), int(f_co[3])))
                        faces.append((int(f_co[0]), int(f_co[3]), int(f_co[2])))
                else:
                    faces.append([int(ver) for ver in f_co])

    points = np.array(points)

    normals = np.array(normals)
    faces = np.array(faces) - 1
    return points, normals, faces



## io
img_size = 1024
dframe=1
bgcolor = None
xtime=1
filedir='database'
vertex_tex=False


gt_animal = 'aardvark_male'

overts_list = []
for i in range(args.nframes):
    # mesh = sr.Mesh.from_obj('/scratch/users/yuefanw/version9/{}/frame_{:06d}.obj'.format(gt_animal,i+1))
    overts,normals,faces = read_obj('/scratch/users/yuefanw/version9/{}/frame_{:06d}.obj'.format(gt_animal, i + 1))


    overts = torch.tensor(overts)[None,:].float().cuda()
    faces = torch.tensor(faces)[None,:].float().cuda()

    overts =overts[:,:,[0,2,1]]
    overts[:,:,1] *= -1


    # pdb.set_trace()
    overts_list.append(overts)

# pdb.set_trace()

proj_mat = torch.eye(4)[np.newaxis].cuda()

cam_list = []
depth_list = []
verts_list = []
verts_pos_list = []

# soft renderer
renderer = sr.SoftRenderer(image_size=img_size, sigma_val=1e-6,
               camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
               light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)

# pdb.set_trace()

for i in range(0,args.nframes):
    overts = overts_list[i]
    # extract camera in/ex
    verts = overts.clone()

    info = np.load('/scratch/users/yuefanw/version9/{}/info/{:04d}.npz'.format(gt_animal,i+1))
    rotmat = torch.tensor(info['cam_rot']).cuda()
    camloc = torch.tensor(info['cam_loc']).cuda()

    bcam2cv = torch.tensor([[1., 0., 0.],
     [0., -1., 0.],
     [0., 0., -1.]]).cuda()

    # pdb.set_trace()

    cam_intrin = torch.tensor(info['intrinsic_mat']).cuda()
    cam_extrin = torch.tensor(info['extrinsic_mat']).cuda()

    # pdb.set_trace()

    fake_rot = torch.matmul(bcam2cv,rotmat.T)

    # pdb.set_trace()
    quat = kornia.rotation_matrix_to_quaternion(fake_rot)

    proj_cam = torch.zeros(1,7).cuda()
    depth = torch.zeros(1,1).cuda()
    proj_cam[:,0]=cam_intrin[0,0]/512   # focal=10
    # proj_cam[:,1] = 1 # x translation = 0
    # proj_cam[:,2] = 1 # y translation = 0

    # pdb.set_trace()
    # camloc_camod = - torch.matmul(rotmat,camloc)

    camloc_camod = cam_extrin[:,-1]

    proj_cam[:,1] = camloc_camod[0]
    proj_cam[:,2] = camloc_camod[1]
    proj_cam[:,3]=quat[3]
    proj_cam[:,4:]=quat[:3]
    # depth[:,0] = 10   # z translation (depth) =10 for spot
    depth[:,0] = camloc_camod[2]

    # cammat = np.asarray(torch.cat([proj_cam[0], depth[0]], 0).cpu())
    # np.savetxt('./database')

    cam_list.append(proj_cam)
    depth_list.append(depth)

    # obj-cam transform 
    Rmat = kornia.quaternion_to_rotation_matrix(torch.cat((proj_cam[:,4:],proj_cam[:,3:4]),1))

    Tmat = torch.cat([proj_cam[:,1:3],depth],1)

    # pdb.set_trace()

    #####   Generate Ground Truth Example
    vertices = verts
    print("Ori vertices ",vertices)

    homo0 = torch.cat((vertices, torch.ones(1,vertices.shape[1], 1).cuda()), dim=2).float()

    # pdb.set_trace()
    verts_raw = torch.matmul(cam_intrin, torch.matmul(cam_extrin, homo0.permute(0, 2, 1))).permute(0, 2, 1)

    verts_proj = torch.matmul(cam_extrin,homo0.permute(0,2,1))
    print("After proj ",verts_raw)
    depth_z = verts_raw[:, :, [2]]
    verts_raw = verts_raw / depth_z
    print("After div depth",verts_raw)
    verts_raw = (verts_raw - 512) / 512
    # verts_raw = verts_raw/512
    print("After norm",verts_raw)
    verts_raw[:, :, 1] *= -1
    print("Final ",verts_raw)


    mesh1 = sr.Mesh(verts_raw, faces)
    rendering_mask = renderer.render_mesh(mesh1)[:, -1]
    rendering_mask = 255 * rendering_mask.detach().cpu().numpy()[0]
    Image.fromarray(rendering_mask.astype(np.uint8)).save('example{}.jpg'.format(i+1))

    pdb.set_trace()

    verts = verts.matmul(Rmat.permute(0,2,1)) + Tmat[:,np.newaxis,:]

    # pdb.set_trace()
    # world_extrin_mat = torch.

    # verts = verts.matmul(Rmat) + Tmat[:,np.newaxis,:]  # obj to cam transform

    verts = torch.cat([verts,torch.ones_like(verts[:, :, 0:1])], dim=-1)
    
    verts_pos_list.append(verts.clone())  # this frame vertex (before projection)
    
    # newmesh = trimesh.Trimesh(vertices=np.asarray(verts[0,:,:3].cpu()), faces=np.asarray(faces[0].cpu()))
   
    # pespective projection: x=fX/Z assuming px=py=0, normalization of Z
    verts[:,:,:-1] = verts[:,:,:-1].clone() * proj_cam[:,:1] / verts[:,:,[2]].clone()

    # verts[:,:,1] *= -1

    mesh2 = sr.Mesh(verts[:,:,:-1], faces)
    rendering_mask = renderer.render_mesh(mesh2)[:, -1]
    rendering_mask = 255 * rendering_mask.detach().cpu().numpy()[0]
    Image.fromarray(rendering_mask.astype(np.uint8)).save('example_new{}.jpg'.format(i+1))

    pdb.set_trace()
    # cammat = np.asarray(torch.cat([proj_cam[0], depth[0]], 0).cpu())
    # np.savetxt('./database')
    # pdb.set_trace()
    #
    # verts[:,:,2] = ( (verts[:,:,2]-verts[:,:,2].min())/(verts[:,:,2].max()-verts[:,:,2].min())-0.5).detach()
    # verts_list.append(verts.clone())
    #
    # # render sil
    # offset = torch.Tensor( renderer.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
    # verts_pre = verts[:,:,:3]-offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
    # if vertex_tex:
    #     rendered = renderer.render_mesh(sr.Mesh(verts_pre,faces,textures=colors[:,:,:3],texture_type='vertex'))
    # else:
    #     rendered = renderer.render_mesh(sr.Mesh(verts_pre,faces,textures=colors,texture_type='surface'))
    # mask_pred=np.asarray(rendered[0,-1,:,:].detach().cpu())
    # img_pred=np.asarray(rendered[0,:3,:,:].permute(1,2,0).detach().cpu())*255
    #
    # if bgcolor is None:
    #     bgcolor = 255-img_pred[mask_pred.astype(bool)].mean(0)
    #
    # img_pred[:,:,::-1][~mask_pred.astype(bool)]=bgcolor[None,::-1]
    # cv2.imwrite('%s/DAVIS/JPEGImages/Full-Resolution/%s/%05d.jpg'     %(filedir,args.outdir,i),img_pred[:,:,::-1])
    # cv2.imwrite('%s/DAVIS/Annotations/Full-Resolution/%s/%05d.png'    %(filedir,args.outdir,i),128*mask_pred)
    # cammat = np.asarray(torch.cat([proj_cam[0], depth[0]],0).cpu())
    # np.savetxt(    '%s/DAVIS/Camera/Full-Resolution/%s/%05d.txt'%(filedir,args.outdir,i),cammat)
    # # newmesh.export('%s/DAVIS/Meshes/Full-Resolution/%s/%05d.obj'%(filedir,args.outdir,i))
