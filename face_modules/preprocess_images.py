# 预处理图像,根据人脸把图像裁剪成256*256的,利用MTCNN得到人脸，然后将其裁剪之后保存。
# 只处理了vggface2中训练集的数据
import torch
from mtcnn import MTCNN
import cv2
import numpy as np

import PIL.Image as Image
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from torchvision import transforms as trans
import os
# import libnvjpeg
# import pickle

# img_root_dir = '/media/taotao/958c7d2d-c4ce-4117-a93b-c8a7aa4b88e3/taotao/part1/'
# save_path = '/media/taotao/958c7d2d-c4ce-4117-a93b-c8a7aa4b88e3/taotao/stars_256_0.85/'
img_root_dir = '/media/gpu/Data2/liuran/images1024x1024/'
save_path = '/media/gpu/Data2/liuran/ffhq_256/'

# embed_path = '/home/taotao/Downloads/celeb-aligned-256/embed.pkl'

device = torch.device('cuda:0')
mtcnn = MTCNN()

model = Backbone(50, 0.6, 'ir_se').to(device)
model.eval()
model.load_state_dict(torch.load('/home/gpu/liuran/FaceShifter/face_modules/model_ir_se50.pth'))

# threshold = 1.54
test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# decoder = libnvjpeg.py_NVJpegDecoder()

ind = 0
embed_map = {}

# 找到图片
for root, dirs, files in os.walk(img_root_dir):
    for name in files:
        # 
        if name.endswith('jpg') or name.endswith('png'):
            try:
                p = os.path.join(root, name)
                # opencv的imread读出来时BGR，需要反向
                img = cv2.imread(p)[:, :, ::-1]
                faces = mtcnn.align_multi(Image.fromarray(img), min_face_size=64, crop_size=(256, 256))
                # if len(faces) == 0:
                if faces == None:
                    continue
                for face in faces:
                    # scaled_img = face.resize((112, 112), Image.ANTIALIAS)
                    # with torch.no_grad():
                    #     embed = model(test_transform(scaled_img).unsqueeze(0).cuda()).squeeze().cpu().numpy()
                    new_path = '%08d.jpg'%ind
                    ind += 1
                    print(new_path)
                    face.save(os.path.join(save_path, new_path))
                # embed_map[new_path] = embed.detach().cpu()
            except Exception as e:
                continue

# with open(embed_path, 'wb') as f:
#     pickle.dump(embed_map, f)
#
# img = cv2.imread('/home/taotao/Pictures/47d947b4d9cf3e2f62c0c8023a1c0dea.jpg')[:,:,::-1]
# # bboxes, faces = mtcnn.align_multi(Image.fromarray(img), limit=10, min_face_size=30)
# bboxes, faces = mtcnn.align(Image.fromarray(img))
# input = test_transform(faces[0]).unsqueeze(0)
# embed = model(input.cuda())
# print(embed.shape)
# print(bboxes)
# face = np.array(faces[0])[:,:,::-1]
# cv2.imshow('', face)
# cv2.waitKey(0)
