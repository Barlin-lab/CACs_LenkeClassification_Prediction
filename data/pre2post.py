from shutil import ReadError
import torch
import os
import csv
from torch.utils import data
import numpy as np
import json

shunxu = {'T1SupAnt': 0, 'T1SupPost': 1, 'T1InfAnt': 2, 'T1InfPost': 3, 'T2SupAnt': 4, 'T2SupPost': 5, 'T2InfAnt': 6, 
    'T2InfPost': 7, 'T3SupAnt': 8, 'T3SupPost': 9, 'T3InfAnt': 10, 'T3InfPost': 11, 'T4SupAnt': 12, 'T4SupPost': 13, 
    'T4InfAnt': 14, 'T4InfPost': 15, 'T5SupAnt': 16, 'T5SupPost': 17, 'T5InfAnt': 18, 'T5InfPost': 19, 'T6SupAnt': 20, 
    'T6SupPost': 21, 'T6InfAnt': 22, 'T6InfPost': 23, 'T7SupAnt': 24, 'T7SupPost': 25, 'T7InfAnt': 26, 'T7InfPost': 27, 
    'T8SupAnt': 28, 'T8SupPost': 29, 'T8InfAnt': 30, 'T8InfPost': 31, 'T9SupAnt': 32, 'T9SupPost': 33, 'T9InfAnt': 34, 
    'T9InfPost': 35, 'T10SupAnt': 36, 'T10SupPost': 37, 'T10InfAnt': 38, 'T10InfPost': 39, 'T11SupAnt': 40, 'T11SupPost': 41, 
    'T11InfAnt': 42, 'T11InfPost': 43, 'T12SupAnt': 44, 'T12SupPost': 45, 'T12InfAnt': 46, 'T12InfPost': 47, 'L1SupAnt': 48, 
    'L1SupPost': 49, 'L1InfAnt': 50, 'L1InfPost': 51, 'L2SupAnt': 52, 'L2SupPost': 53, 'L2InfAnt': 54, 'L2InfPost': 55, 
    'L3SupAnt': 56, 'L3SupPost': 57, 'L3InfAnt': 58, 'L3InfPost': 59, 'L4SupAnt': 60, 'L4SupPost': 61, 'L4InfAnt': 62, 
    'L4InfPost': 63, 'L5SupAnt': 64, 'L5SupPost': 65, 'L5InfAnt': 66, 'L5InfPost': 67, 'S1SupAnt': 68, 'S1SupPost': 69}


class pre2post(torch.utils.data.Dataset):
    def __init__(self, data_root, train=True):
        super(pre2post, self).__init__()


        patient_list = os.listdir(data_root)
        self.patient_ids = []
        self.data = []
        for dir_name in patient_list:
            patient_id = dir_name.split('_')[0]
            self.patient_ids.append(patient_id)
            ap_pre_file = os.path.join(data_root, dir_name, patient_id+'_PRE_AP.json')
            ap_pre = self.load_gt_pts(ap_pre_file)
            self.data.append(ap_pre)


        # print(self.data)
        self.data = np.array(self.data, dtype=float)
        print(self.data.size)
        #处理一下
        for i in range(len(self.data)):
            # self.data[i,0:140:2] -= self.data[i,138]
            # self.data[i,1:140:2] -= self.data[i,139]
            # self.data[i,140::2] -= self.data[i,-2]
            # self.data[i,141::2] -= self.data[i,-1]
            # for j in range(70):
            #     self.data[i][j*2] = self.data[i][j*2] - self.data[i][138]
            #     self.data[i][j*2+1] = self.data[i][j*2+1] - self.data[i][139]
            # for j in range(70, 140):
            #     self.data[i][j*2] = self.data[i][j*2] - self.data[i][-2]
            #     self.data[i][j*2+1] = self.data[i][j*2+1] - self.data[i][-1]
            #归一化
            
            self.data[i,0:140:2] = (self.data[i,0:140:2] - min(self.data[i,0:140:2])) / (max(self.data[i,0:140:2]) - min(self.data[i,0:140:2]))
            self.data[i,1:140:2] = (self.data[i,1:140:2] - min(self.data[i,1:140:2])) / (max(self.data[i,1:140:2]) - min(self.data[i,1:140:2]))
            # self.data[i,140::2] = (self.data[i,140::2] - min(self.data[i,140::2])) / (max(self.data[i,140::2]) - min(self.data[i,140::2]))
            # self.data[i,141::2] = (self.data[i,141::2] - min(self.data[i,141::2])) / (max(self.data[i,141::2]) - min(self.data[i,141::2]))
            # self.data[i,1:140:2] /= max(self.data[i,1:140:2])
            # self.data[i,140::2] /= max(self.data[i,140::2])
            # self.data[i,141::2] /= max(self.data[i,141::2])

        # print(self.data)
        print('total len:{}'.format(len(self.data)))
        

    def __getitem__(self, index):
        return {'points': torch.tensor(self.data[index], dtype=torch.float32), 'patient_id': self.patient_ids[index]}

    def __getitem__(self, index):
        pts1 = []
        for i in range(68):
            pts1.append([self.data[index,2*i], self.data[index,2*i+1]])
        pts1 = np.array(pts1)
        # pts2 = []
        # for i in range(70,138):
            # pts2.append([self.data[index,2*i], self.data[index,2*i+1]])
        # pts2 = np.array(pts2)
        
        # print(pts1.shape)
        center1, hm1 = self.generate_gt(pts1)
        # print("????")
        # center2, hm2 = self.generate_gt(pts2)
        return {'points': [torch.tensor(center1), torch.tensor(hm1)], 'patient_id': self.patient_ids[index]}

    def __len__(self):
        return len(self.data)

    def load_gt_pts(self, annopath):

        with open(annopath, 'r') as f:
            label = json.load(f)
        try:
            points = label['Image Data']['Coronal']['points']
        except:
            print(label['Patient Info'])
            print(annopath)
        points = list(eval(points))
        offset = eval(label ['Image Data']['Coronal']['pos'])
        points = [(point[0]+offset[0], point[1]+offset[1]) for point in points]
        index = eval(label['Image Data']['Coronal']['measureData']['landmarkIndexes'])
        points = self.getlabels(points,index)
        return points

    def getlabels(self, points,index):
        ans = []
        for (key, value) in shunxu.items():
            i = index[key]
            ans.append(points[i][0])
            ans.append(points[i][1])
        return ans

    def generate_gt(self, pts):
        boxes = []
        centers = []
        hm = []
        # print(pts.shape, '###')
        for k in range(0, len(pts), 4):
            pts_4 = pts[k:k+4,:]
            x_inds = np.argsort(pts_4[:, 0])
            pt_l = np.asarray(pts_4[x_inds[:2], :])
            pt_r = np.asarray(pts_4[x_inds[2:], :])
            y_inds_l = np.argsort(pt_l[:,1])
            y_inds_r = np.argsort(pt_r[:,1])
            tl = pt_l[y_inds_l[0], :]
            bl = pt_l[y_inds_l[1], :]
            tr = pt_r[y_inds_r[0], :]
            br = pt_r[y_inds_r[1], :]
            # boxes.append([tl, tr, bl, br])
            boxes.append(tl)
            boxes.append(tr)
            boxes.append(bl)
            boxes.append(br)
            c = np.mean(pts_4, axis=0)
            centers.append(c)
            hm.append(tl-c)
            hm.append(tr-c)
            hm.append(bl-c)
            hm.append(br-c)
        # bboxes = np.asarray(boxes, np.float32) #每个脊椎4个点顺序排好
        # rearrange top to bottom sequence
        # print('########')
        centers = np.asarray(centers, np.float32) 
        hm = np.asarray(hm, np.float32) 
        return centers, hm
    
    