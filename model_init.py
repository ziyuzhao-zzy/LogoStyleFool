import torch
from models import C3D
from vid_model_top_k import I3D_K_Model, C3D_K_Model
from pytorch_i3d import InceptionI3d


def model_initial(model, dataset):
    if model == 'C3D' and dataset == 'UCF101':
        model = C3D(num_classes=101, pretrained=False).cuda()
        checkpoint = torch.load(
            'models/C3D-ucf101.pth.tar',
            map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model = C3D_K_Model(model)
    elif model == 'C3D' and dataset == 'HMDB51':
        model = C3D(num_classes=51, pretrained=False).cuda()
        checkpoint = torch.load(
            'models/C3D-hmdb51.pth.tar',
            map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model = C3D_K_Model(model)
    elif model == 'I3D' and dataset == 'UCF101':
        i3d = InceptionI3d(101, in_channels=3)
        i3d.load_state_dict(torch.load('models/UCF101_I3D.pt'))
        i3d.cuda()
        i3d.train(False)
        i3d.eval()
        model = I3D_K_Model(i3d)
    elif model == 'I3D' and dataset == "HMDB51":
        i3d = InceptionI3d(51, in_channels=3)
        i3d.load_state_dict(torch.load('models/HMDB51_I3D.pt'))
        i3d.cuda()
        i3d.train(False)
        i3d.eval()
        model = I3D_K_Model(i3d)
    return model