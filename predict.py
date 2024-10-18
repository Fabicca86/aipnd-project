import argparse
import os
import random
import json
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model_utils import load_checkpoint, FlowerClassifier

def process_image(image):
    if isinstance(image, torch.Tensor):
        image = T.ToPILImage()(image)
    preprocess = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    return image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if isinstance(image, torch.Tensor):
        image = image.squeeze().cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

def get_random_image(test_dir):
    subdirectories = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    if not subdirectories:
        raise ValueError("No subdirectories found in the specified directory.")
    random_subdirectory = random.choice(subdirectories)
    random_dir_path = os.path.join(test_dir, random_subdirectory)
    print(f'Selected Random Directory: {random_dir_path}')
    image_files = [f for f in os.listdir(random_dir_path) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        raise ValueError("No image files found in the selected directory.")
    random_image_file = random.choice(image_files)
    image_path = os.path.join(random_dir_path, random_image_file)
    image = Image.open(image_path)
    return image, image_path

def predict(image_path, model, topk=5):
    image, image_path = get_random_image(image_path)
    processed_img = process_image(image)
    print(processed_img.shape)
    processed_img = processed_img.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(image_path)
    model.eval()
    with torch.no_grad():
        output = model(processed_img)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_class[0]]
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    top_flowers = [cat_to_name[idx_to_class[idx.item()]] for idx in top_class[0]]
    return top_p[0].tolist(), top_classes, top_flowers

def plot_probabilities(image_path, model, topk=5):
    top_probs, top_classes, top_flowers = predict(image_path, model, topk)
    processed_img, image_path = get_random_image(image_path)  # Obtenha a imagem e o caminho
    processed_img = process_image(processed_img)
    ax = imshow(processed_img, title=top_flowers[0])
    fig, ax = plt.subplots()
    y_pos = np.arange(len(top_flowers))
    ax.barh(y_pos, top_probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_flowers)
    ax.invert_yaxis()
    ax.set_xlabel('Probabilidade')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Classificação de Flores')
    parser.add_argument('--checkpoint', type=str, default='model_checkpoint.pth', help='Caminho para o checkpoint do modelo')
    parser.add_argument('--test_dir', type=str, required=True, help='Caminho para a pasta de teste')
    args = parser.parse_args()
    
    model, optimizer, epoch = load_checkpoint(args.checkpoint)
    plot_probabilities(args.test_dir, model)

if __name__ == "__main__":
    main()
