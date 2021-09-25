import os
import torch.utils.data as data
import data.pre_proc as pre_proc
import cv2
from scipy.io import loadmat
import numpy as np
import json
import pydicom
import torch
from PIL import Image
from torchvision import transforms
import data.custom_transforms as tr

def rearrange_pts(pts):
    boxes = []
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
    return np.asarray(boxes, np.float32)


class BaseDataset_ap(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=4, ):
        super(BaseDataset_ap, self).__init__()
        
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.class_name = ['__background__', 'cell']
        self.num_classes = 68
        # self.img_dir = os.path.join(data_dir, 'data', self.phase)
        # self.img_ids = sorted(os.listdir(self.img_dir))
        # self.img_dir = json.load(os.path.join(data_dir,'split_'+self.phase+'.json')
        # self.img_ids,self.img_dir = self.get_data(data_dir,phase)
        self.patient_ids, self.patient_dir = self.get_data(data_dir,phase)
        self.shunxu = self.get_shunxu()

    def get_data(self, data_dir,phase):
        patient_ids = []
        patient_dir = []
        patient_list = os.listdir(data_dir)
        for file_name in patient_list:
            patient_id = file_name.split('_')[0]
            ap_file = os.path.join(data_dir, file_name, patient_id+'_PRE_AP.dcm')
            bending_l = os.path.join(data_dir, file_name, patient_id+'_BL.dcm')
            bending_r = os.path.join(data_dir, file_name, patient_id+'_BR.dcm')
            lat_file = os.path.join(data_dir, file_name, patient_id+'_PRE_LAT.dcm')
            patient = {'ap':ap_file, 'bending_l':bending_l, 'bending_r':bending_r, 'lat':lat_file}
            patient_dir.append(patient)
            patient_ids.append(patient_id)
        return patient_ids, patient_dir

    def load_images(self, index):
        # image = np.load( image,(1,1,3))
        # image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index]))
        images = []
        for x in ['ap', 'bending_l', 'bending_r', 'lat']:
            img_dir = self.patient_dir[index][x]
            # print(img_dir)
            image = pydicom.read_file(img_dir)
            image = image.pixel_array
            image = self.normalization(image)
            # print(image.shape)
            # image = np.expand_dims(image, 2)
            # image = np.tile(image,(1,1,3))
            images.append(image)
        return images
    
    def normalization(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def load_gt_pts(self, annopath):
        # points = np.load(annopath)
        # pts = []
        # for i in range(68):
        #     pts.append([points[2*i], points[2*i+1]])
        # # pts = loadmat(annopath)['p2']   # num x 2 (x,y)
        # pts = np.array(pts)
        # pts = rearrange_pts(pts)
        with open(os.path.join(self.data_dir, annopath), 'r') as f:
            label = json.load(f)
        try:
            points = label['Image Data'][self.field]['points']
        except:
            print(label['Patient Info'])
            print(annopath)
        points = list(eval(points))
        offset = eval(label ['Image Data'][self.field]['pos'])
        points = [(point[0]+offset[0], point[1]+offset[1]) for point in points]
        index = eval(label['Image Data'][self.field]['measureData']['landmarkIndexes'])
        points = self.getlabels(points,index)

        pts = []
        for i in range(68):
            pts.append([points[2*i], points[2*i+1]])
        # pts = loadmat(annopath)['p2']   # num x 2 (x,y)
        pts = np.array(pts)
        pts = rearrange_pts(pts)

        return pts

    def load_annoFolder(self, img_id):
        return os.path.join(self.data_dir, 'labels', self.phase, img_id)
        # return os.path.join(self.data_dir, 'labels', self.phase, img_id+'.mat')

    def load_annotation(self, index):
        # img_id = self.img_ids[index]
        # annoFolder = self.load_annoFolder(img_id)
        # pts = self.load_gt_pts(annoFolder)
        annoFolder = self.img_dir[index]['json']
        pts = self.load_gt_pts(annoFolder)


        return pts

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        images = self.load_images(index)
    
        out_images = []
        for image in images:
            # print(image.shape)
            out_image = pre_proc.processing_test(image=image, input_h=self.input_h, input_w=self.input_w)
            out_images.append(out_image)
        out_images = torch.stack(out_images)
        return {'images': out_images, 'patient_id': patient_id}
        

    def __len__(self):
        return len(self.patient_ids)

    
    def get_shunxu(self):
        jizhui = ['T', 'L']
        weizhi = ['SupAnt','SupPost','InfAnt','InfPost']
        shunxu = {}
        num = 0
        for i in range(12):
            for s in weizhi:
                
                index = 'T' + str(i+1) + s
                shunxu[index] = num
                num = num + 1
        for i in range(5):
            for s in weizhi:
                
                index = 'L' + str(i+1) + s
                shunxu[index] = num
                num = num + 1      
        shunxu['S1SupAnt'] = num
        num = num + 1
        shunxu['S1SupPost'] = num
        num = num + 1
        return shunxu

    def getlabels(self, points,index):
        ans = []
        for (key, value) in self.shunxu.items():
            i = index[key]
            ans.append(points[i][0])
            ans.append(points[i][1])
        return ans


class BaseDataset_lat(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=4):
        super(BaseDataset_lat, self).__init__()
        
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.class_name = ['__background__', 'cell']
        self.num_classes = 68
        # self.img_dir = os.path.join(data_dir, 'data', self.phase)
        # self.img_ids = sorted(os.listdir(self.img_dir))
        # self.img_dir = json.load(os.path.join(data_dir,'split_'+self.phase+'.json')
        # self.img_ids,self.img_dir = self.get_data(data_dir,phase)
        self.patient_ids, self.patient_dir = self.get_data(data_dir,phase)
        self.shunxu = self.get_shunxu()

    def get_data(self, data_dir,phase):
        # if phase=='train':
        #     with open(os.path.join(data_dir,'split_train.json'),'r') as f:
        #         split = json.load(f)
        # else:
        #     with open(os.path.join(data_dir,'split_test.json'),'r') as f:
        #         split = json.load(f)
        
        patient_ids = []
        patient_dir = []

        # for patient in split:
        #     if patient['ap'] and patient['bending_l'] and patient['bending_r'] and patient['lat']:
        #         patient_dir.append(patient)
        #         patient_ids.append(patient['id'])

        patient_list = os.listdir(data_dir)
        for file_name in patient_list:
            patient_id = file_name.split('_')[0]
            lat_file = os.path.join(data_dir, file_name, patient_id+'_PRE_LAT.dcm')
            patient_dir.append(lat_file)
            patient_ids.append(patient_id)
        return patient_ids, patient_dir
        
            
        
        # return patient_ids, patient_dir

    def load_images(self, index):
        # image = np.load( image,(1,1,3))
        # image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index]))
        
        image = pydicom.read_file(self.patient_dir[index])
        image = image.pixel_array
        # print(image.shape)
        image = self.normalization(image)
        # image = np.expand_dims(image, 2)
        # image = np.tile(image,(1,1,3))
        return image
    
    def normalization(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def load_gt_pts(self, annopath):
        # points = np.load(annopath)
        # pts = []
        # for i in range(68):
        #     pts.append([points[2*i], points[2*i+1]])
        # # pts = loadmat(annopath)['p2']   # num x 2 (x,y)
        # pts = np.array(pts)
        # pts = rearrange_pts(pts)
        with open(os.path.join(self.data_dir, annopath), 'r') as f:
            label = json.load(f)
        try:
            points = label['Image Data'][self.field]['points']
        except:
            print(label['Patient Info'])
            print(annopath)
        points = list(eval(points))
        offset = eval(label ['Image Data'][self.field]['pos'])
        points = [(point[0]+offset[0], point[1]+offset[1]) for point in points]
        index = eval(label['Image Data'][self.field]['measureData']['landmarkIndexes'])
        points = self.getlabels(points,index)

        pts = []
        for i in range(68):
            pts.append([points[2*i], points[2*i+1]])
        # pts = loadmat(annopath)['p2']   # num x 2 (x,y)
        pts = np.array(pts)
        pts = rearrange_pts(pts)

        return pts

    def load_annoFolder(self, img_id):
        return os.path.join(self.data_dir, 'labels', self.phase, img_id)
        # return os.path.join(self.data_dir, 'labels', self.phase, img_id+'.mat')

    def load_annotation(self, index):
        # img_id = self.img_ids[index]
        # annoFolder = self.load_annoFolder(img_id)
        # pts = self.load_gt_pts(annoFolder)
        annoFolder = self.img_dir[index]['json']
        pts = self.load_gt_pts(annoFolder)


        return pts

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        image = self.load_images(index)
        # print(image)
        # print(image.shape)
        # out_images = []

        out_image = pre_proc.processing_test(image=image, input_h=self.input_h, input_w=self.input_w)
        # out_images = torch.stack(out_images)
        return {'images': out_image, 'patient_id': patient_id}
        

    def __len__(self):
        return len(self.patient_ids)

    
    def get_shunxu(self):
        jizhui = ['T', 'L']
        weizhi = ['SupAnt','SupPost','InfAnt','InfPost']
        shunxu = {}
        num = 0
        for i in range(12):
            for s in weizhi:
                
                index = 'T' + str(i+1) + s
                shunxu[index] = num
                num = num + 1
        for i in range(5):
            for s in weizhi:
                
                index = 'L' + str(i+1) + s
                shunxu[index] = num
                num = num + 1      
        shunxu['S1SupAnt'] = num
        num = num + 1
        shunxu['S1SupPost'] = num
        num = num + 1
        return shunxu

    def getlabels(self, points,index):
        ans = []
        for (key, value) in self.shunxu.items():
            i = index[key]
            ans.append(points[i][0])
            ans.append(points[i][1])
        return ans

class BaseDataset_csvl(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=4):
        super(BaseDataset_csvl, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.class_name = ['__background__', 'cell']
        self.num_classes = 68
        # self.img_dir = os.path.join(data_dir, 'data', self.phase)
        # self.img_ids = sorted(os.listdir(self.img_dir))
        # self.img_dir = json.load(os.path.join(data_dir,'split_'+self.phase+'.json')
        self.patient_ids, self.patient_dir = self.get_data(data_dir,phase)
        self.shunxu = self.get_shunxu()
        print('total samples:{}'.format(len(self.patient_dir)))


    def get_data(self, data_dir,phase):
        # if phase=='train':
        #     with open(os.path.join(data_dir,'split_train.json'),'r') as f:
        #         split = json.load(f)
        # else:
        #     with open(os.path.join(data_dir,'split_test.json'),'r') as f:
        #         split = json.load(f)
        
        patient_ids = []
        patient_dir = []

        # for patient in split:
        #     if patient['ap'] and patient['bending_l'] and patient['bending_r'] and patient['lat']:
        #         patient_dir.append(patient)
        #         patient_ids.append(patient['id'])

        patient_list = os.listdir(data_dir)
        for file_name in patient_list:
            patient_id = file_name.split('_')[0]
            lat_file = os.path.join(data_dir, file_name, patient_id+'_PRE_AP.dcm')
            patient_dir.append(lat_file)
            patient_ids.append(patient_id)
        return patient_ids, patient_dir

    # def load_image(self, index):
    #     # image = np.load(os.path.join(self.img_dir, self.img_ids[index]))
    #     # image = np.expand_dims(image, 2)
    #     # image = np.tile(image,(1,1,3))
    #     image = pydicom.read_file(os.path.join(self.data_dir, self.img_dir[index]['dcm']))
    #     image = image.pixel_array
    #     #直接归一化
    #     image = self.normalization(image)
    #     image = np.expand_dims(image, 2)
    #     image = np.tile(image,(1,1,3))
    #     #图像灰度级不统一
    #     return image

    def load_image(self, index):
        # image = np.load( image,(1,1,3))
        # image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index]))
        
        image = pydicom.read_file(self.patient_dir[index])
        image = image.pixel_array
        # print(image.shape)
        image = self.normalization(image)
        # image = np.expand_dims(image, 2)
        # image = np.tile(image,(1,1,3))
        return image
    
    def normalization(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range


    def load_gt_pts(self, annopath):
        # points = np.load(annopath)
        # pts = []
        # for i in range(68):
        #     pts.append([points[2*i], points[2*i+1]])
        # # pts = loadmat(annopath)['p2']   # num x 2 (x,y)
        # pts = np.array(pts)
        # pts = rearrange_pts(pts)
        with open(os.path.join(self.data_dir, annopath), 'r') as f:
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

        pts = []
        for i in range(68):
            pts.append([points[2*i], points[2*i+1]])
        # pts = loadmat(annopath)['p2']   # num x 2 (x,y)
        pts = np.array(pts)
        pts = rearrange_pts(pts)

        return pts

    def load_annoFolder(self, img_id):
        return os.path.join(self.data_dir, 'labels', self.phase, img_id)
        # return os.path.join(self.data_dir, 'labels', self.phase, img_id+'.mat')

    def load_annotation(self, index):
        # img_id = self.img_ids[index]
        # annoFolder = self.load_annoFolder(img_id)
        # pts = self.load_gt_pts(annoFolder)
        annoFolder = self.img_dir[index]['json']
        pts = self.load_gt_pts(annoFolder)


        return pts

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=1000),
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        image = self.load_image(index)
        
        oimage = image * 255
        # print(oimage.shape)
        sample = {'image': Image.fromarray(oimage.astype(np.uint8)), 'label': Image.fromarray(oimage.astype(np.uint8))}
        sample = self.transform_val(sample)



        #数据处理只要保证训练集和测试集统一
        if self.phase == 'test':
            images = pre_proc.processing_test(image=image, input_h=self.input_h, input_w=self.input_w)
            return {'images': images, 'patient_id': patient_id, 'oimage': sample['image']}
        else:
            aug_label = False
            if self.phase == 'train':
                aug_label = True
            pts = self.load_annotation(index)   # num_obj x h x w
            #预处理图片和标签
            out_image, pts_2 = pre_proc.processing_train(image=image,
                                                         pts=pts,
                                                         image_h=self.input_h, #输入的大小
                                                         image_w=self.input_w,
                                                         down_ratio=self.down_ratio,
                                                         aug_label=aug_label,
                                                         img_id=img_id)

            data_dict = pre_proc.generate_ground_truth(image=out_image,
                                                       pts_2=pts_2,
                                                       image_h=self.input_h//self.down_ratio, #结果的大小
                                                       image_w=self.input_w//self.down_ratio,
                                                       img_id=img_id)
            return data_dict

    def __len__(self):
        return len(self.patient_ids)

    
    def get_shunxu(self):
        jizhui = ['T', 'L']
        weizhi = ['SupAnt','SupPost','InfAnt','InfPost']
        shunxu = {}
        num = 0
        for i in range(12):
            for s in weizhi:
                
                index = 'T' + str(i+1) + s
                shunxu[index] = num
                num = num + 1
        for i in range(5):
            for s in weizhi:
                
                index = 'L' + str(i+1) + s
                shunxu[index] = num
                num = num + 1      
        shunxu['S1SupAnt'] = num
        num = num + 1
        shunxu['S1SupPost'] = num
        num = num + 1
        return shunxu

    def getlabels(self, points,index):
        ans = []
        for (key, value) in self.shunxu.items():
            i = index[key]
            ans.append(points[i][0])
            ans.append(points[i][1])
        return ans