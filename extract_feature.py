import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from utils import yaml_config_hook
from torchvision.datasets import ImageFolder


def inference(loader, simclr_model, device):
    feature_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        
        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector


def rename_keys(state_dict, prefix='module.'):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(prefix, '')  # Remove the 'module.' prefix
        new_state_dict[new_key] = value

    return new_state_dict


parser = argparse.ArgumentParser(description="SimCLR")
config = yaml_config_hook("./config/config.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))
    
args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ImageFolder(
            '/home/zach/C101/train',
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        )
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
encoder = get_resnet(args.resnet, pretrained=False)
n_features = encoder.fc.in_features  # get dimensions of fc layer

# load pre-trained model from checkpoint
simclr_model = SimCLR(encoder, args.projection_dim, n_features)
model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
renamed_state_dict = rename_keys(torch.load(model_fp, map_location=args.device.type))

simclr_model.load_state_dict(renamed_state_dict, strict=False)
simclr_model = simclr_model.to(args.device)
simclr_model.eval()

print("### Creating features from pre-trained context model ###")
train_x = inference(train_loader, simclr_model, args.device)

# 存储特征到 pickle 文件
output_pickle_path = 'output_features_10000.pkl'
with open(output_pickle_path, 'wb') as f:
    pickle.dump(train_x, f)

print(f"Features extracted and saved to {output_pickle_path}")