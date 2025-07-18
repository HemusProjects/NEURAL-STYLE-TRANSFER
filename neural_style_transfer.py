import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

def load_image(image_path):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

content_img = load_image("content.jpg")
style_img = load_image("style.jpg")
generated_img = content_img.clone().requires_grad_(True)

vgg = models.vgg19(pretrained=True).features.to(device).eval()

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
                  '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

content_features = get_features(content_img, vgg)
style_features = get_features(style_img, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

content_weight = 1e4
style_weight = 1e2

optimizer = optim.Adam([generated_img], lr=0.003)

for i in range(500):
    generated_features = get_features(generated_img, vgg)
    content_loss = torch.mean((generated_features['conv4_2'] - content_features['conv4_2'])**2)
    style_loss = 0
    for layer in style_grams:
        gen_feature = generated_features[layer]
        gen_gram = gram_matrix(gen_feature)
        style_gram = style_grams[layer]
        layer_style_loss = torch.mean((gen_gram - style_gram)**2)
        style_loss += layer_style_loss
    total_loss = content_weight * content_loss + style_weight * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

output = generated_img.clone().squeeze()
output = output.detach().cpu().numpy()
output = output.transpose(1, 2, 0).astype("uint8")

plt.imshow(output)
plt.axis("off")
plt.show()
