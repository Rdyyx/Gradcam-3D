import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import Dataset  # using your own dataset
from grad_cam import GradCAM
from torch import nn
import torch.nn.functional as F
from models.resnet_50 import resnet50_3d # import your model


def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

def visualize_comparison(args, input_data, feature_map, slice_idx, decoder):
    """三维可视化对比函数"""
    # 获取原始切片
    original_slices = {
        'x': input_data[0, 0, slice_idx, :, :].numpy(),
        'y': input_data[0, 0, :, slice_idx, :].numpy(),
        'z': input_data[0, 0, :, :, slice_idx].numpy()
    }

    # 特征图预处理
    if isinstance(feature_map, np.ndarray):
        feature_map = torch.tensor(feature_map, dtype=torch.float32)
        feature_map=feature_map.unsqueeze(0)
    # 三维插值到原始尺寸
    upsampled_feature = F.interpolate(
        feature_map,
        size=tuple(input_data.shape[2:]),
        mode='trilinear',
        align_corners=False
    ).squeeze()

    # 创建可视化画布
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plt.set_cmap('jet')

    # 遍历三个轴向
    for idx, axis in enumerate(['x', 'y', 'z']):
        # 原始切片
        axes[idx, 0].imshow(original_slices[axis], cmap='gray')
        axes[idx, 0].set_title(f'Original {axis.upper()}-Slice {slice_idx}')
        # 特征图切片
        if axis == 'x':
            feature_slice = upsampled_feature[slice_idx, :, :]
        elif axis == 'y':
            feature_slice = upsampled_feature[:, slice_idx, :]
        else:
            feature_slice = upsampled_feature[:, :, slice_idx]

        # 归一化
        feature_slice = (feature_slice - feature_slice.min()) / (feature_slice.max() - feature_slice.min())

        # 叠加可视化
        # axes[idx, 1].imshow(original_slices[axis], cmap='gray')
        axes[idx, 1].imshow(original_slices[axis], cmap='gray')
        im = axes[idx, 1].imshow(feature_slice, alpha=0.5)
        plt.colorbar(im, ax=axes[idx, 1])
        axes[idx, 1].set_title(f'Feature Map {axis.upper()}-Slice {slice_idx}')
    plt.tight_layout()
    img_save_path = args['img_save_path']
    # plt.show()
    plt.savefig(f'{img_save_path}\{decoder}slice_idx{slice_idx}.png')

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)

            x = module(x)
            print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x

        return outputs

def main():

    path = 'config.yaml'
    args = read_yaml(path)

    model = resnet50_3d()
    # 替换对应权重
    weight_path= args['weight_path']
    weights = torch.load(weight_path, map_location=torch.device(args['device']))
    # model.load_state_dict(weights['model_state_dict'], strict=False)
    model.load_state_dict(weights, strict=False)
    model.eval()

    # load image
    val_dataset = Dataset(args['data_path'])
    val_loader = DataLoader(
        val_dataset,
        batch_size=args['batch_size'],
        shuffle=args['shuffle'],
        pin_memory=args['pin_memory'],
        num_workers=args['num_workers'],
    )
    for batch_idx, batch in enumerate(val_loader):
        imgs, label = batch
        break

    input_data = imgs   # [1,1,64,64,64]
    input_data = torch.tensor(input_data, dtype=torch.float32)
    target_layers = [model.layer1]
    print(target_layers)


    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    target_category = 0

    grayscale_cam = cam(input_tensor=input_data, target_category=target_category)

    decoder = args['table_name']
    for slice_idx in range(args['start_slice'],args['end_slice']):
        visualize_comparison(args, input_data, grayscale_cam, slice_idx, decoder)
if __name__ == '__main__':
    main()
