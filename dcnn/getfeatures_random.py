import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import h5py

vgg = models.vgg16(pretrained=False)

print(vgg)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
)

#matrix_ly1 = np.zeros([450, 64*224*224])
matrix_ly2 = np.zeros([450, 64*112*112])
#matrix_ly3 = np.zeros([450, 128*112*112])
matrix_ly4 = np.zeros([450, 128*56*56])
#matrix_ly5 = np.zeros([450, 256*56*56])
matrix_ly6 = np.zeros([450, 256*56*56])
#matrix_ly7 = np.zeros([450, 256*28*28])
matrix_ly8 = np.zeros([450, 512*28*28])
#matrix_ly9 = np.zeros([450, 512*28*28])
matrix_ly10 = np.zeros([450, 512*14*14])
#matrix_ly11 = np.zeros([450, 512*14*14])
matrix_ly12 = np.zeros([450, 512*14*14])
#matrix_ly13 = np.zeros([450, 512*7*7])
matrix_ly14 = np.zeros([450, 4096])
#matrix_ly15 = np.zeros([450, 4096])
matrix_ly16 = np.zeros([450, 1000])

conditions = ["f", "u", "s"]

index = 0
for con in conditions:
    for i in range(150):

        filename = "../stimuli/stimuli_"+con+str(i+1).zfill(3)+".bmp"
        x = Image.open(filename).convert("RGB")
        im = transform(x)
        im.unsqueeze_(dim=0)
        vgg1 = nn.Sequential(*list(vgg.features.children())[:5])
        ly2 = vgg1(im).data.numpy()
        activation_ly2 = np.reshape(ly2, [64*112*112])

        vgg1 = nn.Sequential(*list(vgg.features.children())[:10])
        ly4 = vgg1(im).data.numpy()
        activation_ly4 = np.reshape(ly4, [128*56*56])

        vgg1 = nn.Sequential(*list(vgg.features.children())[:14])
        ly6 = vgg1(im).data.numpy()
        activation_ly6 = np.reshape(ly6, [256*56*56])

        vgg1 = nn.Sequential(*list(vgg.features.children())[:19])
        ly8 = vgg1(im).data.numpy()
        activation_ly8 = np.reshape(ly8, [512*28*28])

        vgg1 = nn.Sequential(*list(vgg.features.children())[:24])
        ly10 = vgg1(im).data.numpy()
        activation_ly10 = np.reshape(ly10, [512*14*14])

        vgg1 = nn.Sequential(*list(vgg.features.children())[:28])
        ly12 = vgg1(im).data.numpy()
        activation_ly12 = np.reshape(ly12, [512*14*14])

        vgg1 = nn.Sequential(*list(vgg.features.children())[:])
        output1 = vgg1(im)
        output1np = output1.data.numpy()
        output1np = np.reshape(output1np, [1, 512*7*7])
        output1 = torch.from_numpy(output1np)
        vgg2 = nn.Sequential(*list(vgg.classifier.children())[:3])
        ly14 = vgg2(output1).data.numpy()
        activation_ly14 = np.reshape(ly14, [4096])

        vgg2 = nn.Sequential(*list(vgg.classifier.children())[:])
        ly16 = vgg2(output1).data.numpy()
        activation_ly16 = np.reshape(ly16, [1000])

        #matrix_ly1[index] = activation_ly1
        matrix_ly2[index] = activation_ly2
        #matrix_ly3[index] = activation_ly3
        matrix_ly4[index] = activation_ly4
        #matrix_ly5[index] = activation_ly5
        matrix_ly6[index] = activation_ly6
        #matrix_ly7[index] = activation_ly7
        matrix_ly8[index] = activation_ly8
        #matrix_ly9[index] = activation_ly9
        matrix_ly10[index] = activation_ly10
        #matrix_ly11[index] = activation_ly11
        matrix_ly12[index] = activation_ly12
        #matrix_ly13[index] = activation_ly13
        matrix_ly14[index] = activation_ly14
        #matrix_ly15[index] = activation_ly15
        matrix_ly16[index] = activation_ly16

        index = index + 1

        print(con, "-", str(i+1))

f = h5py.File("features_random/ly2.h5", "w")
f.create_dataset("activations", data=matrix_ly2)
f.close()
f = h5py.File("features_random/ly4.h5", "w")
f.create_dataset("activations", data=matrix_ly4)
f.close()
f = h5py.File("features_random/ly6.h5", "w")
f.create_dataset("activations", data=matrix_ly6)
f.close()
f = h5py.File("features_random/ly8.h5", "w")
f.create_dataset("activations", data=matrix_ly8)
f.close()
f = h5py.File("features_random/ly10.h5", "w")
f.create_dataset("activations", data=matrix_ly10)
f.close()
f = h5py.File("features_random/ly12.h5", "w")
f.create_dataset("activations", data=matrix_ly12)
f.close()
f = h5py.File("features_random/ly14.h5", "w")
f.create_dataset("activations", data=matrix_ly14)
f.close()
f = h5py.File("features_random/ly16.h5", "w")
f.create_dataset("activations", data=matrix_ly16)
f.close()
