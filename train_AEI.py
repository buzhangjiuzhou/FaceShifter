from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import time
import os
import torchvision
import cv2
from apex import amp
import visdom
from torchsummaryX import summary

from utils.show import make_image, get_grid_image
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from utils.Dataset import FaceEmbed, Faces, cat_dataloaders
from utils.loss import dssim, hinge_loss


vis = visdom.Visdom(server='127.0.0.1', env='faceshifter-modify', port=8097)
batch_size = 4
lr_G = 4e-4
lr_D = 4e-4
lr_id = 5e-4
max_epoch = 2000
show_step = 10
save_epoch = 1
optim_level = 'O1'
data_flag = "ffhq"

data_aligned = '/media/gpu/Data2/liuran/ffhq_256/'
src_path = './data_src'
dst_path = './data_dst'

model_save_path = './saved_models/'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# GAN训练部分
G = AEI_Net(c_id=512)
D = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d)

G.train()
D.train()

G = G.cuda()
D = D.cuda()

summary(G, torch.zeros(1, 3, 256, 256), torch.zeros(1, 512))


opt_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0, 0.999))


G, opt_G = amp.initialize(G, opt_G, opt_level=optim_level)
D, opt_D = amp.initialize(D, opt_D, opt_level=optim_level)

try:
    G.load_state_dict(torch.load('./saved_models/G_latest.pth'), strict=False)
    D.load_state_dict(torch.load('./saved_models/D_latest.pth'), strict=False)
except Exception as e:
    print(e)

# arcface部分
arcface = Backbone(50, 0.6, 'ir_se')
arcface.eval()
arcface = arcface.cuda()

arcface.load_state_dict(torch.load('./face_modules/model_ir_se50.pth'), strict=False)

MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()


for epoch in range(0, max_epoch):

    # 每一轮次重新load 数据
    if data_flag == "ffhq":
        dataset = FaceEmbed([data_aligned], same_prob=0.1)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        data_length = len(dataloader)
    else:
        dataset_src = Faces(src_path)
        dataloader_src = DataLoader(dataset_src, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        dataset_dst = Faces(dst_path)
        dataloader_dst = DataLoader(dataset_dst, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

        dataloader = cat_dataloaders([dataloader_src, dataloader_dst], batch_size)
        data_length = max(len(dataloader_src), len(dataloader_dst))

    for iteration, data in enumerate(dataloader):

        if data_flag == "ffhq":
            image_src, image_dst, same_person = data
            same_person = same_person.cuda()
        else :
            if data == None:
                break
            image_src, image_dst = data
            if image_src is None or image_dst is None:
                break
                

        start_time = time.time()

        image_src = image_src.cuda()
        image_dst = image_dst.cuda()
        
        # embed = embed.to(device)
        with torch.no_grad():
            embed, image_src_feats = arcface(F.interpolate(image_src[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))

        # train G
        opt_G.zero_grad()

        Y, image_dst_attr = G(image_dst, embed)

        Di = D(Y)
        L_adv = 0

        for di in Di:
            L_adv += hinge_loss(di[0], True)
        

        Y_aligned = Y[:, :, 19:237, 19:237]
        ZY, Y_feats = arcface(F.interpolate(Y_aligned, [112, 112], mode='bilinear', align_corners=True))
        L_id =(1 - torch.cosine_similarity(embed, ZY, dim=1)).mean()

        Y_attr = G.get_attr(Y)
        L_attr = 0
        for i in range(len(image_dst_attr)):
            L_attr += torch.mean(torch.pow(image_dst_attr[i] - Y_attr[i], 2).reshape(batch_size, -1), dim=1).mean()
        L_attr /= 2.0

        L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - image_dst, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

        lossG = 1*L_adv + 10*L_attr + 5*L_id + 10*L_rec

        with amp.scale_loss(lossG, opt_G) as scaled_loss:
            scaled_loss.backward(retain_graph=True)

        opt_G.step()

        # train D
        opt_D.zero_grad()

        fake_D = D(Y.detach())
        loss_fake = 0
        for di in fake_D:
            loss_fake += hinge_loss(di[0], False)

        true_D = D(image_src)
        loss_true = 0
        for di in true_D:
            loss_true += hinge_loss(di[0], True)

        lossD = 0.5*(loss_true.mean() + loss_fake.mean())

        with amp.scale_loss(lossD, opt_D) as scaled_loss:
            scaled_loss.backward()

        opt_D.step()

        batch_time = time.time() - start_time

        if iteration % show_step == 0:
            image = make_image(image_src, image_dst, Y)
            vis.image(image[::-1, :, :], opts={'title': 'result'}, win='result')
            cv2.imwrite('./gen_images/latest.jpg', image.transpose([1,2,0]))
        
        vis.line(X=np.array([iteration + (epoch * data_length)]), Y=np.array([lossD.item()]), win='loss D', opts={'title': 'loss D'}, update='append')
        vis.line(X=np.array([iteration + (epoch * data_length)]), Y=np.array([lossG.item()]), win='loss G', opts={'title': 'loss G'}, update='append')
        vis.line(X=np.array([iteration + (epoch * data_length)]), Y=np.array([L_adv.item()]), win='L_adv', opts={'title': 'L_adv'}, update='append')
        vis.line(X=np.array([iteration + (epoch * data_length)]), Y=np.array([L_id.item()]), win='L_id', opts={'title': 'L_id'}, update='append')
        vis.line(X=np.array([iteration + (epoch * data_length)]), Y=np.array([L_attr.item()]), win='L_attr', opts={'title': 'L_attr'}, update='append')
        vis.line(X=np.array([iteration + (epoch * data_length)]), Y=np.array([L_rec.item()]), win='L_rec', opts={'title': 'L_rec'}, update='append')
 
        print(f'epoch: {epoch}    {iteration} / {data_length}')
        print(f'lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s')
        print(f'L_adv: {L_adv.item()} L_id: {L_id.item()} L_attr: {L_attr.item()} L_rec: {L_rec.item()}')
        
        if iteration % 1000 == 0:
            torch.save(G.state_dict(), './saved_models/G_latest.pth')
            torch.save(D.state_dict(), './saved_models/D_latest.pth')
            print("model saved!")


