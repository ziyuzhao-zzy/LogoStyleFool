import os
import torch
from libs.Matrix import MulLayer
from libs.models import encoder4
from libs.models import decoder4
from torchvision.transforms import Resize
from transformlogo import logoset


os.environ["OMP_NUM_THREADS"] = "1"
cuda = torch.cuda.is_available()

vgg = encoder4().cuda()
dec = decoder4().cuda()
matrix = MulLayer("r41").cuda()
vgg.load_state_dict(torch.load('models/vgg_r41.pth'))
dec.load_state_dict(torch.load('models/dec_r41.pth'))
matrix.load_state_dict(torch.load('models/r41.pth', map_location='cuda'))


def style_perturb_logo(logo_index, style_img, h, w):
    contentV = logoset[logo_index].unsqueeze(0).cuda()
    styleV = style_img.unsqueeze(0).cuda()
    transfer_size = Resize((224, 224))
    contentV = transfer_size(contentV)
    styleV = transfer_size(styleV)
    transfer_size = Resize((h, w))
    with torch.no_grad():
        sF = vgg(styleV)
        cF = vgg(contentV)
        feature, transmatrix = matrix(cF["r41"], sF["r41"])
        transfer = dec(feature)
    transfer = transfer.clamp(0,1)
    transfer = transfer_size(transfer)
    transfer = transfer.squeeze(0)  # remove the fake batch dimension
    return transfer