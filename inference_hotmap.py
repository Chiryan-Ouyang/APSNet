import os
import torch
import cv2
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
from models import APSnet18, APSnet34, APSnet50
from train import default_loader
import yaml
import argparse

def apply_colormap(heatmap, colormap=cv2.COLORMAP_HSV):
    heatmap = 1 - heatmap
    heatmap = np.power(heatmap, 2) 
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_map = cv2.applyColorMap(heatmap, colormap)
    colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)
    
    return  colored_map 

def get_cam(target_layer, model, image):
    model.eval()
    activations = []

    def hook(module, input, output):
        activations.append(output)

    hook_handle = target_layer.register_forward_hook(hook)
    with torch.no_grad():
        model(image)
    hook_handle.remove()

    activation_map = activations[0][0, 0].cpu().numpy()  

    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    return activation_map

def infer_image(model, image_path, output_dir, img_mean, img_std):
    img = default_loader(image_path, True)
    img_gray = img.convert('L')  
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[img_mean], std=[img_std])
    ])
    img_tensor = transform(img_gray).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    model = model.to(device) 

    with torch.no_grad():
        model.eval()
        out_class, out_seg = model(img_tensor)
        predicted_label = out_class.argmax(dim=1).cpu().item()
        segmentation_mask = F.sigmoid(out_seg).cpu().squeeze().numpy()

        target_layer = model.outconv

        cam = get_cam(target_layer, model, img_tensor)

        cam_colored = apply_colormap(cam)
        cam_img = Image.fromarray(cam_colored)
        
        cam_img = cam_img.resize(img.size, Image.BILINEAR)
        cam_output_path = os.path.join(output_dir, f'{os.path.basename(image_path)}_cam.png')
        cam_img.save(cam_output_path)

        return predicted_label, segmentation_mask, cam
def main(config):
    model_name = config['model']  
    if model_name == 'APSnet18':
        model = APSnet18(use_se=config['use_se'], use_dual_path=config['use_dual_path'], block_type=config['block_type'])
    elif model_name == 'APSnet34':
        model = APSnet34(use_se=config['use_se'], use_dual_path=config['use_dual_path'], block_type=config['block_type'])
    elif model_name == 'APSnet50':
        model = APSnet50(use_se=config['use_se'], use_dual_path=config['use_dual_path'], block_type=config['block_type'])
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(config['model_weights'], 'train300models.pth')  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for filename in os.listdir(config['input_dir']):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(config['input_dir'], filename)
            predicted_label, segmentation_mask, cam = infer_image(model, image_path, config['output_dir'], config['img_mean'], config['img_std'])

            # Save the results
            output_label_file = os.path.join(config['output_dir'], f'{filename}_label.txt')
            with open(output_label_file, 'w') as f:
                f.write(str(predicted_label))

            output_seg_file = os.path.join(config['output_dir'], f'{filename}_seg.png')
            Image.fromarray((segmentation_mask * 255).astype(np.uint8)).save(output_seg_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='APSNet Inference')
    parser.add_argument('--config', type=str, default='config.yaml', help='path to the yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    if 'cuda_devices' in config:
        os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_devices']
    img_mean = config.get('img_mean', 0.5)  # Use config values
    img_std = config.get('img_std', 0.5)
    
    # Set the directory containing the original images
    input_dir = config.get('input_dir', './datasets/')  # Use config values
    
    # Create the output directory if it doesn't exist
    output_dir = config.get('output_dir', './output/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(config)