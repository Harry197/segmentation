import os
import json
import argparse

import torch
import torch.onnx
import torchvision.models as models

from deeplabv3 import DeepLabv3Plus


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
parser.add_argument('--classes_path', required=True)
parser.add_argument('--export_path', type=str, default='./export/')
args = parser.parse_args()

os.makedirs(args.export_path, exist_ok=True)

if os.path.isdir(args.checkpoint_path):
    file_list = os.listdir(args.checkpoint_path)
    full_list = [os.path.join(args.checkpoint_path, i) for i in file_list]
    model_path = sorted(full_list, key=os.path.getmtime)[-1]
else:
    model_path = args.checkpoint_path

with open(args.classes_path) as json_file:
    color_dict = json.load(json_file)
num_classes = len(color_dict)

model = DeepLabv3Plus(models.resnet50(pretrained=True),
                      num_classes=num_classes+1)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

batch_size = 1

x = torch.randn(batch_size, 3, 384, 384, requires_grad=True)
torch_out = model(x)

torch.onnx.export(model,
                  x,
                  os.path.join(args.export_path, "model.onnx"),
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}}
                 )
