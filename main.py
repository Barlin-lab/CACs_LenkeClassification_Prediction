from genericpath import exists
import os
import re
from posixpath import join
from sys import path
import numpy as np
from numpy.lib.histograms import _ravel_and_check_weights
from models import spinal_net
import torch
from decoder import DecDecoder
from data.dataset import BaseDataset_ap, BaseDataset_lat, BaseDataset_csvl
import cv2
from cobb_evaluate import *
from data.pre2post import pre2post
from models import regression
from torch.autograd import Variable
import matplotlib.pyplot as plt
from modeling.deeplab import *
import data.custom_transforms as tr
from PIL import Image
from torchvision import transforms
import cv2 as cv

def load_model(model, resume):
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    model.load_state_dict(state_dict_, strict=False)
    return model

def ap_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = {'hm': 1,
            'reg': 2*1,
            'wh': 2*4,}

    model_ap = spinal_net.SpineNet(heads=heads,
                                    pretrained=True,
                                    down_ratio=4,
                                    final_kernel=1,
                                    head_conv=256)

    model_lat = spinal_net.SpineNet(heads=heads,
                                    pretrained=True,
                                    down_ratio=4,
                                    final_kernel=1,
                                    head_conv=256)


    # self.num_classes = args.num_classes
    decoder = DecDecoder(K=17, conf_thresh=0.2)
    # dataset = BaseDataset

    ap_save_path = './models/pretrain/ap_model_best.pth'
    model_ap = load_model(model_ap, ap_save_path)
    model_ap = model_ap.to(device)
    model_ap.eval()

    lat_save_path = './models/pretrain/lat_model_best.pth'
    model_lat = load_model(model_lat, lat_save_path)
    model_lat = model_lat.to(device)
    model_lat.eval()

    # dataset_module = dataset
    dsets = BaseDataset_ap(data_dir='./data/DATAEXPORT_TEST/',
                            phase='test',
                            input_h=1024,
                            input_w=512,
                            down_ratio=4
                            )

    data_loader = torch.utils.data.DataLoader(dsets,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,
                                            pin_memory=True)


    for cnt, data_dict in enumerate(data_loader):
        images = data_dict['images'][0]
        patient_id = data_dict['patient_id'][0]
        images = images.to(device)
        print("###############################################")
        print('processing {}/{} patient ... {}'.format(cnt, len(data_loader), patient_id))

        index = ['ap', 'bending_l', 'bending_r', 'lat']
        angles = []
        pts_all = []
        ori_image_points_all = []
        for i in range(4):
        
            print('processing image {}'.format(index[i]))
            if i == 3:
                model = model_lat
            else:
                model = model_ap
            #####################处理 ap#################
            # print(images.shape)
            with torch.no_grad():
                output = model(images[i])
                hm = output['hm']
                wh = output['wh']
                reg = output['reg']

            torch.cuda.synchronize(device)
            pts2 = decoder.ctdet_decode(hm, wh, reg)   # 17, 11
            pts0 = pts2.copy()
            pts0[:,:10] *= 4

            # print('totol pts num is {}'.format(len(pts2)))

            ori_images = dsets.load_images(dsets.patient_ids.index(patient_id))
            ori_image = ori_images[i]
            k = 256
            ori_image = ori_image * k


            ori_image_regress = cv2.resize(ori_image, (512, 1024))
            ori_image_points = ori_image_regress.copy()

            h,w,c = ori_image.shape
            pts0 = np.asarray(pts0, np.float32)
            # pts0[:,0::2] = pts0[:,0::2]/args.input_w*w
            # pts0[:,1::2] = pts0[:,1::2]/args.input_h*h
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]

            # ori_image_regress, ori_image_points = draw_points.draw_landmarks_regress_test(pts0,
            #                                                                             ori_image_regress,
            #                                                                             ori_image_points)

            pts3 = []
            for p in pts0:
                point1 = [p[2], p[3]]
                point2 = [p[4], p[5]]
                point3 = [p[6], p[7]]
                point4 = [p[8], p[9]]
                pts3.extend([point1,point2,point3,point4])
            pts3 = np.array(pts3)            
            pts_all.append(pts3)
            ori_image_points_all.append(ori_image_points)

        angles, ptss, image = cal_angles(pts_all[0], ori_image_points_all[0])

        if not os.path.exists(os.path.join('./results/', patient_id)):
            os.makedirs(os.path.join('./results/', patient_id))
        cv2.imwrite(os.path.join('./results/', patient_id, index[0]+'.jpg'), image)
        
        report = open(os.path.join('./results', patient_id, 'report.txt'), 'a')
        

        ptss2 = []
        for x in ptss:
            ptss2.append((x[0]+1,x[1]+1))
        print('三段脊椎角度和对应脊椎段：\n',angles, ptss2, file=report)

        ################处理bending, lat################
        scoliosis_type = []
        scoliosis_seg = ['PT', 'MT', 'TL']
        for i, angle in enumerate(angles):
            print("-----{}段诊断流程".format(scoliosis_seg[i]), file=report)
            if angle < 25:
                print("#####{}为非结构弯：脊椎{}-{}：{}".format(scoliosis_seg[i], ptss[i][0]+1, ptss[i][1]+1, round(angles[i],2)), file=report)
                scoliosis_type.append(False)
            else:
                angle1, image1 = cal_angles_with_pos(pts_all[1], ori_image_points_all[1], ptss[i])
                angle2, image2 = cal_angles_with_pos(pts_all[2], ori_image_points_all[2], ptss[i])

                angle_bending = angle1 if angle1 < angle2 else angle2
                image = image1 if angle1 < angle2 else image2
                i_lr = 1 if angle1 < angle2 else 2
                cv2.imwrite(os.path.join('./results/', patient_id, scoliosis_seg[i]+'_'+index[i_lr]+'.jpg'), image)

                if angle_bending > 25:
                    print("#####{}为结构弯：直立位{}-{}：{}，bending位：{}".format(scoliosis_seg[i],ptss[i][0]+1, ptss[i][1]+1, round(angles[i],2), round(angle_bending,2)), file=report)
                    scoliosis_type.append(True)
                else:
                    if i == 0:
                        pos = (1, 4)
                    else:
                        pos = (9, 13)
                    angle_lat, image = cal_angles_with_pos(pts_all[3], ori_image_points_all[3], pos)
                    cv2.imwrite(os.path.join('./results/', patient_id, scoliosis_seg[i]+'_'+index[3]+'.jpg'), image)
                    if angle_lat < 20:
                        print("#####{}为非结构弯：脊椎{}-{}：{}，bending位：{}，矢状面：{}-{}：{}".format(scoliosis_seg[i],ptss[i][0]+1, ptss[i][1]+1, round(angles[i],2), round(angle_bending,2), pos[0],pos[1],round(angle_lat,2)), file=report)
                        scoliosis_type.append(False)
                    else:
                        print("#####{}为结构弯：脊椎{}-{}：{}，bending位：{}，矢状面：{}-{}：{}".format(scoliosis_seg[i],ptss[i][0]+1, ptss[i][1]+1, round(angles[i],2), round(angle_bending,2), pos[0],pos[1],round(angle_lat,2)), file=report)
                        scoliosis_type.append(True) 

def ap_report_v2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = {'hm': 1,
            'reg': 2*1,
            'wh': 2*4,}
    print(device)

    model_ap = spinal_net.SpineNet(heads=heads,
                                    pretrained=True,
                                    down_ratio=4,
                                    final_kernel=1,
                                    head_conv=256)

    model_lat = spinal_net.SpineNet(heads=heads,
                                    pretrained=True,
                                    down_ratio=4,
                                    final_kernel=1,
                                    head_conv=256)


    # self.num_classes = args.num_classes
    decoder = DecDecoder(K=17, conf_thresh=0.2)
    # dataset = BaseDataset

    ap_save_path = './models/pretrain/ap_model_best.pth'
    model_ap = load_model(model_ap, ap_save_path)
    model_ap = model_ap.to(device)
    model_ap.eval()

    lat_save_path = './models/pretrain/lat_model_best.pth'
    model_lat = load_model(model_lat, lat_save_path)
    model_lat = model_lat.to(device)
    model_lat.eval()

    # dataset_module = dataset
    dsets = BaseDataset_ap(data_dir='./data/DATAEXPORT_TEST/',
                            phase='test',
                            input_h=1024,
                            input_w=512,
                            down_ratio=4
                            )

    data_loader = torch.utils.data.DataLoader(dsets,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,
                                            pin_memory=True)


    for cnt, data_dict in enumerate(data_loader):
        images = data_dict['images'][0]
        patient_id = data_dict['patient_id'][0]
        images = images.to(device)
        print("###############################################")
        print('processing {}/{} patient ... {}'.format(cnt, len(data_loader), patient_id))

        index = ['ap', 'bending_l', 'bending_r', 'lat']
        angles = []
        pts_all = []
        ori_image_points_all = []
        for i in range(4):
        
            print('processing image {}'.format(index[i]))
            if i == 3:
                model = model_lat
            else:
                model = model_ap
            #####################处理 ap#################
            # print(images.shape)
            with torch.no_grad():
                output = model(images[i])
                hm = output['hm']
                wh = output['wh']
                reg = output['reg']

            # torch.cuda.synchronize(device)
            pts2 = decoder.ctdet_decode(hm, wh, reg)   # 17, 11
            pts0 = pts2.copy()
            pts0[:,:10] *= 4

            # print('totol pts num is {}'.format(len(pts2)))

            ori_images = dsets.load_images(dsets.patient_ids.index(patient_id))
            ori_image = ori_images[i]
            k = 256
            ori_image = ori_image * k


            ori_image_regress = cv2.resize(ori_image, (512, 1024))
            ori_image_points = ori_image_regress.copy()

            h,w,c = ori_image.shape
            pts0 = np.asarray(pts0, np.float32)
            # pts0[:,0::2] = pts0[:,0::2]/args.input_w*w
            # pts0[:,1::2] = pts0[:,1::2]/args.input_h*h
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]

            # ori_image_regress, ori_image_points = draw_points.draw_landmarks_regress_test(pts0,
            #                                                                             ori_image_regress,
            #                                                                             ori_image_points)

            pts3 = []
            for p in pts0:
                point1 = [p[2], p[3]]
                point2 = [p[4], p[5]]
                point3 = [p[6], p[7]]
                point4 = [p[8], p[9]]
                pts3.extend([point1,point2,point3,point4])
            pts3 = np.array(pts3)            
            pts_all.append(pts3)
            ori_image_points_all.append(ori_image_points)

        angles, ptss, image = cal_angles(pts_all[0], ori_image_points_all[0])

        

        if not os.path.exists(os.path.join('./results/', patient_id)):
            os.makedirs(os.path.join('./results/', patient_id))
        cv2.imwrite(os.path.join('./results/', patient_id, index[0]+'.jpg'), image)
        
        report = open(os.path.join('./results', patient_id, 'report.txt'), 'a')
        

        ptss2 = []
        for x in ptss:
            ptss2.append((x[0]+1,x[1]+1))
        print('三段脊椎角度和对应脊椎段：\n',angles, ptss2, file=report)

        t2t5angle_lat, t2t5image = cal_angles_with_pos(pts_all[3], ori_image_points_all[3].copy(), (1,4))
        cv2.imwrite(os.path.join('./results/', patient_id, 'lat_t2_t5.jpg'), t2t5image)
        t10l2angle_lat, t10l2image = cal_angles_with_pos(pts_all[3], ori_image_points_all[3].copy(), (9,13))
        cv2.imwrite(os.path.join('./results/', patient_id, 'lat_t10_l2.jpg'), t10l2image)
        t5t12angle_lat, t5t12image = cal_angles_with_pos(pts_all[3], ori_image_points_all[3].copy(), (4,11))
        cv2.imwrite(os.path.join('./results/', patient_id, 'lat_t5_t12.jpg'), t5t12image)
        ################处理bending, lat################
        scoliosis_type = []
        scoliosis_seg = ['PT', 'MT', 'TL']
        for i, angle in enumerate(angles):
            print("-----{}段诊断流程".format(scoliosis_seg[i]), file=report)
            bl_angle, bl_image = cal_angles_with_pos(pts_all[1], ori_image_points_all[1].copy(), ptss[i])
            br_angle, br_image = cal_angles_with_pos(pts_all[2], ori_image_points_all[2].copy(), ptss[i])
            cv2.imwrite(os.path.join('./results/', patient_id, scoliosis_seg[i]+'_bl.jpg'), bl_image)
            cv2.imwrite(os.path.join('./results/', patient_id, scoliosis_seg[i]+'_br.jpg'), br_image)
            angle_bending = bl_angle if bl_angle < br_angle else br_angle
            j1 = segType_Trans(ptss[i][0] + 1)
            j2 = segType_Trans(ptss[i][1] + 1)
            if i == 0:
                pos = ("T2", "T5")
                angle_lat = t2t5angle_lat
            else:
                pos = ("T10", "L2")
                angle_lat = t10l2angle_lat

            if angle < 25:
                print("#####{}为非结构弯：脊椎{}-{}：{}，bending位：{}，矢状面：{}-{}：{}".format(scoliosis_seg[i],j1, j2, round(angles[i],2), round(angle_bending,2), pos[0],pos[1],round(angle_lat,2)), file=report)
                scoliosis_type.append(False)
            else:
                # angle1, image1 = cal_angles_with_pos(pts_all[1], ori_image_points_all[1], ptss[i])
                # angle2, image2 = cal_angles_with_pos(pts_all[2], ori_image_points_all[2], ptss[i])


                # image = bl_image if bl_angle < br_angle else br_image
                # i_lr = 1 if bl_angle < br_angle else 2
                
                if angle_bending > 25:
                    print("#####{}为结构弯：脊椎{}-{}：{}，bending位：{}，矢状面：{}-{}：{}".format(scoliosis_seg[i],j1, j2, round(angles[i],2), round(angle_bending,2), pos[0],pos[1],round(angle_lat,2)), file=report)
                    scoliosis_type.append(True)
                else:

                    # angle_lat, image = cal_angles_with_pos(pts_all[3], ori_image_points_all[3], pos)
                    # cv2.imwrite(os.path.join('./results/', patient_id, scoliosis_seg[i]+'_'+index[3]+'.jpg'), image)
                    
                    if angle_lat < 20:
                        print("#####{}为非结构弯：脊椎{}-{}：{}，bending位：{}，矢状面：{}-{}：{}".format(scoliosis_seg[i],j1, j2, round(angles[i],2), round(angle_bending,2), pos[0],pos[1],round(angle_lat,2)), file=report)
                        scoliosis_type.append(False)
                    else:
                        print("#####{}为结构弯：脊椎{}-{}：{}，bending位：{}，矢状面：{}-{}：{}".format(scoliosis_seg[i],j1, j2, round(angles[i],2), round(angle_bending,2), pos[0],pos[1],round(angle_lat,2)), file=report)
                        scoliosis_type.append(True) 
        
def lat_report():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    heads = {'hm': 1,
                'reg': 2*1,
                'wh': 2*4,}

    model = spinal_net.SpineNet(heads=heads,
                                        pretrained=True,
                                        down_ratio=4,
                                        final_kernel=1,
                                        head_conv=256)
    # num_classes = args.num_classes
    decoder = DecDecoder(K=17, conf_thresh=0.2)
    # dataset = BaseDataset
    lat_save_path = './models/pretrain/lat_model_best.pth'
    model = load_model(model, lat_save_path)
    model = model.to(device)
    model.eval()

    # dataset_module = dataset
    dsets = BaseDataset_lat(data_dir='./data/DATAEXPORT_TEST',
                            phase='test',
                            input_h=1024,
                            input_w=512,
                            down_ratio=4
                            )

    data_loader = torch.utils.data.DataLoader(dsets,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=1,
                                                pin_memory=True)


    for cnt, data_dict in enumerate(data_loader):
        image = data_dict['images'][0]
        patient_id = data_dict['patient_id'][0]
        image = image.to(device)
        print("###############################################")
        print('processing {}/{} patient ... {}'.format(cnt, len(data_loader), patient_id))

        # print(image.shape)
        with torch.no_grad():
            output = model(image)
            hm = output['hm']
            wh = output['wh']
            reg = output['reg']

        # torch.cuda.synchronize(device)
        pts2 = decoder.ctdet_decode(hm, wh, reg)   # 17, 11
        pts0 = pts2.copy()
        pts0[:,:10] *= 4

        # print('totol pts num is {}'.format(len(pts2)))

        ori_image = dsets.load_images(dsets.patient_ids.index(patient_id))
        # ori_image = ori_images[3]
        # print(ori_image.shape)
        k = 256
        ori_image = ori_image * k


        ori_image_regress = cv2.resize(ori_image, (512, 1024))
        ori_image_points = ori_image_regress.copy()

        h,w,c = ori_image.shape
        pts0 = np.asarray(pts0, np.float32)
        # pts0[:,0::2] = pts0[:,0::2]/args.input_w*w
        # pts0[:,1::2] = pts0[:,1::2]/args.input_h*h
        sort_ind = np.argsort(pts0[:,1])
        pts0 = pts0[sort_ind]

        # ori_image_regress, ori_image_points = draw_points.draw_landmarks_regress_test(pts0,
        #                                                                             ori_image_regress,
        #                                                                             ori_image_points)
        if not os.path.exists(os.path.join('./results', patient_id)):
            os.mkdir(os.path.join('./results', patient_id))
        report = open(os.path.join('./results', patient_id, 'report.txt'), 'a')

        pts3 = []
        for p in pts0:
            point1 = [p[2], p[3]]
            point2 = [p[4], p[5]]
            point3 = [p[6], p[7]]
            point4 = [p[8], p[9]]
            pts3.extend([point1,point2,point3,point4])
        pts3 = np.array(pts3) 
        print(pts3.shape)           
        pos = (4, 11)
        angles, image = cal_angles_with_pos(pts3, ori_image_points, pos)
        # print(image.shape)
        if not os.path.exists('./results'):
            os.mkdir('./results/')

        cv2.imwrite(os.path.join('./results/', patient_id, 'lat.jpg'), image)
        print('T5-T12:{}'.format(round(angles,2)), file=report)
        if angles < 10:
            print('矢状面： -类', file=report)
        elif angles < 40:
            print('矢状面： N类', file=report)
        else:
            print('矢状面： +类', file=report)

def csvl_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = {'hm': 1,
            'reg': 2*1,
            'wh': 2*4,}

    model_ap = spinal_net.SpineNet(heads=heads,
                                    pretrained=True,
                                    down_ratio=4,
                                    final_kernel=1,
                                    head_conv=256)

    # model_lat = spinal_net.SpineNet(heads=heads,
    #                                 pretrained=True,
    #                                 down_ratio=4,
    #                                 final_kernel=1,
    #                                 head_conv=256)


    # self.num_classes = args.num_classes
    decoder = DecDecoder(K=18, conf_thresh=0.2)
    # dataset = BaseDataset

    ap_save_path = './models/pretrain/s1.pth'
    model_ap = load_model(model_ap, ap_save_path)
    model_ap = model_ap.to(device)
    model_ap.eval()


    model_csvl = DeepLab(num_classes=2,
                            backbone='resnet',
                            output_stride=16,
                            sync_bn=None,
                            freeze_bn=False)
    csvl_save_path = './models/pretrain/csvl_model_best.pth.tar'
    model_csvl = load_model(model_csvl, csvl_save_path)
    model_csvl = model_csvl.to(device)
    model_csvl.eval()

    dsets = BaseDataset_csvl(data_dir='./data/DATAEXPORT_TEST/',
                            phase='test',
                            input_h=1024,
                            input_w=512,
                            down_ratio=4
                            )

    data_loader = torch.utils.data.DataLoader(dsets,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,
                                            pin_memory=True)


    print("--load data done--")
    for cnt, data_dict in enumerate(data_loader):
        images = data_dict['images'][0]
        patient_id = data_dict['patient_id'][0]
        oimage = data_dict['oimage']
        # print(images.shape, oimage.shape)
        oimage = oimage.to(device)
        images = images.to(device)
        print('processing {}/{} image ... {}'.format(cnt, len(data_loader), patient_id))

        if not os.path.exists(os.path.join('./results', patient_id)):
            os.mkdir(os.path.join('./results', patient_id))
        report = open(os.path.join('./results', patient_id, 'report.txt'), 'a')
        path = os.path.join('./results', patient_id, 'report.txt')
        with torch.no_grad():
            output = model_ap(images)
            hm = output['hm']
            wh = output['wh']
            reg = output['reg']

        # torch.cuda.synchronize(device)
        pts2 = decoder.ctdet_decode(hm, wh, reg)   # 17, 11
        pts0 = pts2.copy()
        pts0[:,:10] *= 4

        print('totol pts num is {}'.format(len(pts2)))

        ori_image = dsets.load_image(dsets.patient_ids.index(patient_id))
        k = 256 / 1024
        ori_image = ori_image * k


        ori_image_regress = cv2.resize(ori_image, (512, 1024))
        ori_image_points = ori_image_regress.copy()

        h,w,c = ori_image.shape
        pts0 = np.asarray(pts0, np.float32)
        # pts0[:,0::2] = pts0[:,0::2]/args.input_w*w
        # pts0[:,1::2] = pts0[:,1::2]/args.input_h*h
        sort_ind = np.argsort(pts0[:,1])
        pts0 = pts0[sort_ind]

        pts3 = []
        for p in pts0:
            point1 = [p[2], p[3]]
            point2 = [p[4], p[5]]
            point3 = [p[6], p[7]]
            point4 = [p[8], p[9]]
            pts3.extend([point1,point2,point3,point4])
        pts3 = np.array(pts3)

        jieduan = int(min(pts3[52][1], pts3[53][1]))
        sly = (pts3[-4][1] - jieduan) *1000/ (images.shape[2] - jieduan)
        sry = (pts3[-3][1] - jieduan) *1000/ (images.shape[2] - jieduan)
        slx =  (pts3[-4][0]/images.shape[3]) * 1000
        srx =  (pts3[-3][0]/images.shape[3]) * 1000

        jieduan = int((jieduan/images.shape[2]) * oimage.shape[2])
        csvl = int((slx + srx) / 2)

        oimage = oimage[:,:,jieduan:,:]
        oimage = oimage[0]
        simage = oimage.data
        oimage = oimage.data.cpu().numpy().transpose(1,2,0)

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=1000),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        sample = {'image': Image.fromarray(oimage.astype(np.uint8)), 'label': Image.fromarray(oimage.astype(np.uint8))}
        sample = composed_transforms(sample)
        oimage = sample['image'].to(device)

        oimage = oimage.unsqueeze(0)
        # print(oimage)

        with torch.no_grad():
            output = model_csvl(oimage)
        result = output.data.cpu().numpy()
        result = np.argmax(result, axis=1)
        result = result.squeeze()
        result = result * 255
        result = np.clip(result, 0, 255)
        result = np.array(result,np.uint8)

        
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel, iterations=5)
        
        
        # result = result[0]
        # png_file = patient_id + '.png'
        png_file = os.path.join('./results', patient_id, 'csvl_seg.png')
        output = Image.fromarray(result.astype(np.uint8)).convert('P')

        simage = oimage.data.cpu().squeeze()
        std = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).repeat(1,1000,1000)
        mu = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).repeat(1,1000,1000)
        simage = simage * std + mu
        simage = simage.numpy()
        simage = np.transpose(simage, (1,2,0))
        simage = simage * 255
        simage = np.clip(simage, 0, 255)
        simage = np.array(simage,np.uint8)

        ret,thresh=cv2.threshold(result,127,255,0)

        # temp = np.ones(result.shape,np.uint8)*255
        contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        cv2.drawContours(simage,contours,-1,(0,255,0),5)

        # simage = np.expand_dims(simage,2).repeat(3,2)

        cv2.line(simage,
                 (int(csvl), int(0)),
                 (int(csvl), int(simage.shape[0])),
                 color=(0, 255, 255), thickness=2, lineType=2)  
        cv2.line(simage,
                 (int(slx), int(sly)),
                 (int(srx), int(sry)),
                 color=(0, 255, 255), thickness=2, lineType=2)  


        

        left, right = get_csvl_ans(result, csvl)
        
        cv2.line(simage,
                 (left[1], left[0]),
                 (right[1],right[0]),
                 color=(255, 0, 0), thickness=2, lineType=2)
        if left[1] < csvl:   # 1, <
            if right[1] < csvl:  # 1, <
                print('csvl类型： C type', file=report)
            else:
                print('csvl类型： B type', file=report)
        else:
            print('csvl类型： A type',file=report)



        cv2.imwrite(png_file, simage)
        report.close()
        final_lenke = lenke_judge(path)
        with open(r'' + path, 'a') as f:
            f.write(final_lenke)
            f.close()

def get_csvl_ans(mask, csvl):
    pts = np.argwhere(mask ==  255)
    pts = [[x[0], x[1]-csvl] for x in pts]
    pts = sorted(pts, key=lambda x: -abs(x[1]))
    max_pt = pts[0]

    max_pt[1] += csvl

    right = max_pt[:]
    left = max_pt[:]
    que = []
    que.append(max_pt)
    dir = [[0,1],[0,-1],[1,0],[-1,0]]
    while len(que) > 0:
        p = que.pop(0)
        for i in range(4):
            new_p = [p[0]+dir[i][0], p[1]+dir[i][1]]
            if new_p[0] < mask.shape[0] and new_p[0] >= 0 and new_p[1] < mask.shape[1] and new_p[1] >= 0 and mask[new_p[0],new_p[1]] == 255:
                que.append(new_p)
                if new_p[1] > right[1]:
                    right = new_p[:]
                if new_p[1] < left[1]:
                    left = new_p[:]
                mask[new_p[0],new_p[1]] = 0

    return left, right


def post_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = {'hm': 1,
             'reg': 2 * 1,
             'wh': 2 * 4, }

    model_ap = spinal_net.SpineNet(heads=heads,
                                   pretrained=True,
                                   down_ratio=4,
                                   final_kernel=1,
                                   head_conv=256)

    # self.num_classes = args.num_classes
    decoder = DecDecoder(K=17, conf_thresh=0.2)
    # dataset = BaseDataset

    ap_save_path = './models/pretrain/ap_model_best.pth'
    model_ap = load_model(model_ap, ap_save_path)
    model_ap = model_ap.to(device)
    model_ap.eval()

    # dataset_module = dataset
    dsets = BaseDataset_ap(data_dir='./data/DATAEXPORT_TEST/',
                           phase='test',
                           input_h=1024,
                           input_w=512,
                           down_ratio=4
                           )

    data_loader = torch.utils.data.DataLoader(dsets,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True)

    model = regression.regression(input_dim=34).to(device)
    checkpoint = torch.load('./models/pretrain/post_model_best.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    for cnt, data_dict in enumerate(data_loader):
        images = data_dict['images'][0]
        patient_id = data_dict['patient_id'][0]
        images = images.to(device)
        print("###############################################")
        print('processing {}/{} patient ... {}'.format(cnt, len(data_loader), patient_id))

        #####################处理 ap#################
        with torch.no_grad():
            output = model_ap(images[0])
            hm = output['hm']
            wh = output['wh']
            reg = output['reg']

        # torch.cuda.synchronize(device)
        pts2 = decoder.ctdet_decode(hm, wh, reg)  # 17, 11
        pts0 = pts2.copy()
        pts0[:, :10] *= 4

        # h,w,c = ori_image.shape
        pts0 = np.asarray(pts0, np.float32)
        # pts0[:,0::2] = pts0[:,0::2]/args.input_w*w
        # pts0[:,1::2] = pts0[:,1::2]/args.input_h*h
        sort_ind = np.argsort(pts0[:, 1])
        pts0 = pts0[sort_ind]

        pts3 = []
        for p in pts0:
            point1 = [p[2], p[3]]
            point2 = [p[4], p[5]]
            point3 = [p[6], p[7]]
            point4 = [p[8], p[9]]
            pts3.extend([point1, point2, point3, point4])
        pts3 = np.array(pts3)  # 68 * 2

        pts3[:, 0] = (pts3[:, 0] - min(pts3[:, 0])) / (max(pts3[:, 0]) - min(pts3[:, 0]))
        pts3[:, 1] = (pts3[:, 1] - min(pts3[:, 1])) / (max(pts3[:, 1]) - min(pts3[:, 1]))

        center1, hm1 = generate_gt(pts3)
        center1 = torch.tensor(center1)
        center1 = center1.to(device)

        center1 = Variable(center1)
        center1 = center1.unsqueeze(0)
        # print(center1)
        pre_center = model(center1)

        result_dir = os.path.join('./results', patient_id)
        pre = center1[0].cpu().data.view(17, 2).numpy()
        post = pre_center[0].cpu().view(17, 2).data.numpy()

        pre[:, 1] = (1 - pre[:, 1]) * 4.5  # 调整脊椎的长宽比，影响侧弯的角度
        # print(pre)
        post[:, 1] = (1 - post[:, 1]) * 6

        # pre[:,0] = -pre[:,0]
        # post[:,0] = -post[:,0]

        plt.figure()
        plt.xlim(0, 1)
        plt.ylim(0, 4)
        plt.subplot(131)
        plt.axis('equal')
        # plt.xticks([]),plt.yticks([])
        # 拟合曲线
        z1 = np.polyfit(pre[:, 1], pre[:, 0], 40)
        p1 = np.poly1d(z1)
        yvals = p1(pre[:, 1])
        plt.plot(yvals, pre[:, 1], 'grey', linewidth=15, marker="o", markersize=5, markerfacecolor="white")

        plt.subplot(133)
        plt.xlim(-0.5, 1.5)
        # plt.ylim(0,2)
        # plt.xticks([]),plt.yticks([])
        # print(post.shape)

        z1 = np.polyfit(post[:, 1], post[:, 0], 40)
        p1 = np.poly1d(z1)
        yvals = p1(post[:, 1])
        plt.plot(yvals, post[:,1], 'grey', linewidth=15, marker="o", markersize=5, markerfacecolor="white")

        plt.savefig(os.path.join(result_dir, 'post_operation.jpg'))


def generate_gt(pts):
    boxes = []
    centers = []
    hm = []
    # print(pts.shape, '###')
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k + 4, :]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:, 1])
        y_inds_r = np.argsort(pt_r[:, 1])
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
        hm.append(tl - c)
        hm.append(tr - c)
        hm.append(bl - c)
        hm.append(br - c)
    # bboxes = np.asarray(boxes, np.float32) #每个脊椎4个点顺序排好
    # rearrange top to bottom sequence
    # print('########')
    centers = np.asarray(centers, np.float32)
    hm = np.asarray(hm, np.float32)
    return centers, hm

def segType_Trans(num):
    segType = dict(
        zip(["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4",
             "L5"], list(range(1, 18))))
    for j in list(segType.values()):
        if num == j:
            ptss_sup=list(segType.keys())[list(segType.values()).index(j)]
        continue
    return ptss_sup
def lenke_judge(path):
    lenke_type = ["Lenke1", "Lenke2", "Lenke3", "Lenke4", "Lenke5", "Lenke6"]
    report=open(r''+path)
    a_list = report.readlines()
    pt = re.search(pattern=r"(\u975e?)\u7ed3\u6784\u5f2f", string=a_list[5].encode('utf8').decode('utf8')).group()
    mt = re.search(pattern=r"(\u975e?)\u7ed3\u6784\u5f2f", string=a_list[7].encode('utf8').decode('utf8')).group()
    tl = re.search(pattern=r"(\u975e?)\u7ed3\u6784\u5f2f", string=a_list[9].encode('utf8').decode('utf8')).group()
    pt_angle = float(re.search(pattern="([TL][0-9]+-[TL][0-9]+)：(\d+.\d+)", string=a_list[5]).group(2)) #.strip('，bending'))
    mt_angle = float(re.search(pattern="([TL][0-9]+-[TL][0-9]+)：(\d+.\d+)", string=a_list[7]).group(2)) #.strip('，bending'))
    tl_angle = float(re.search(pattern="([TL][0-9]+-[TL][0-9]+)：(\d+.\d+)", string=a_list[9]).group(2)) #.strip('，bending'))
    sag_mod = re.search(pattern=r"(.)\u7c7b", string=a_list[1].encode('utf8').decode('utf8')).group().strip('类')
    csvl_mod = re.search(pattern="([A-Z])", string=a_list[10]).group()
    if pt=="非结构弯" and mt == "结构弯" and tl == "非结构弯" :
        label_type= lenke_type[0]
    elif pt=="结构弯" and mt == "结构弯" and tl == "非结构弯" :
        label_type = lenke_type[1]
    elif pt=="非结构弯" and mt == "结构弯" and tl == "结构弯" and mt_angle > tl_angle:
        label_type = lenke_type[2]
    elif pt=="结构弯" and mt == "结构弯" and tl == "结构弯" :
        label_type = lenke_type[3]
    elif pt=="非结构弯" and mt == "非结构弯" and tl == "结构弯" :
        label_type = lenke_type[4]
    elif pt=="非结构弯" and mt == "结构弯" and tl == "结构弯" and mt_angle < tl_angle:
        label_type = lenke_type[5]
    else:
        label_type="None lenke type"
    return label_type+csvl_mod+sag_mod



if __name__ == '__main__':
    
    print("################################generate lat report####################################")
    lat_report()
    print("################################generate ap report####################################")
    ap_report_v2()
    print("################################generate csvl report##### ###############################")
    csvl_report()
    print("################################generate post operation report####################################")
    post_report()



