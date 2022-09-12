import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import h5py

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = F.relu(self.conv1_1(x))
        ly1 = x.data.numpy()
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2, 2)
        ly2 = x.data.numpy()
        x = F.relu(self.conv2_1(x))
        ly3 = x.data.numpy()
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2, 2)
        ly4 = x.data.numpy()
        x = F.relu(self.conv3_1(x))
        ly5 = x.data.numpy()
        x = F.relu(self.conv3_2(x))
        ly6 = x.data.numpy()
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, 2, 2)
        ly7 = x.data.numpy()
        x = F.relu(self.conv4_1(x))
        ly8 = x.data.numpy()
        x = F.relu(self.conv4_2(x))
        ly9 = x.data.numpy()
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, 2, 2)
        ly10 = x.data.numpy()
        x = F.relu(self.conv5_1(x))
        ly11 = x.data.numpy()
        x = F.relu(self.conv5_2(x))
        ly12 = x.data.numpy()
        x = F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, 2, 2)
        ly13 = x.data.numpy()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        ly14 = x.data.numpy()
        # x = F.dropout(x, 0.5)
        x = F.relu(self.fc7(x))
        ly15 = x.data.numpy()
        ly16 = self.fc8(x).data.numpy()
        # x = F.dropout(x, 0.5)
        return ly1, ly2, ly3, ly4, ly5, ly6, ly7, ly8, ly9, ly10, ly11, ly12, ly13, ly14, ly15, ly16

net = VGG_16()
pthfile = "vgg_face_dag.pth"
net.load_state_dict(torch.load(pthfile))
print(net)

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
matrix_ly16 = np.zeros([450, 2622])

conditions = ["f", "u", "s"]

index = 0
for con in conditions:
    for i in range(150):

        filename = "../stimuli/stimuli_"+con+str(i+1).zfill(3)+".bmp"
        x = Image.open(filename).convert("RGB")
        im = transform(x)
        im.unsqueeze_(dim=0)
        ly1, ly2, ly3, ly4, ly5, ly6, ly7, ly8, ly9, ly10, ly11, ly12, ly13, ly14, ly15, ly16 = net(im)
        #activation_ly1 = np.reshape(ly1, [64*224*224])
        activation_ly2 = np.reshape(ly2, [64*112*112])
        #activation_ly3 = np.reshape(ly3, [128*112*112])
        activation_ly4 = np.reshape(ly4, [128*56*56])
        #activation_ly5 = np.reshape(ly5, [256*56*56])
        activation_ly6 = np.reshape(ly6, [256*56*56])
        #activation_ly7 = np.reshape(ly7, [256*28*28])
        activation_ly8 = np.reshape(ly8, [512*28*28])
        #activation_ly9 = np.reshape(ly9, [512*28*28])
        activation_ly10 = np.reshape(ly10, [512*14*14])
        #activation_ly11 = np.reshape(ly11, [512*14*14])
        activation_ly12 = np.reshape(ly12, [512*14*14])
        #activation_ly13 = np.reshape(ly13, [512*7*7])
        activation_ly14 = np.reshape(ly14, [4096])
        #activation_ly15 = np.reshape(ly15, [4096])
        activation_ly16 = np.reshape(ly16, [2622])
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

f = h5py.File("features/ly2.h5", "w")
f.create_dataset("activations", data=matrix_ly2)
f.close()
f = h5py.File("features/ly4.h5", "w")
f.create_dataset("activations", data=matrix_ly4)
f.close()
f = h5py.File("features/ly6.h5", "w")
f.create_dataset("activations", data=matrix_ly6)
f.close()
f = h5py.File("features/ly8.h5", "w")
f.create_dataset("activations", data=matrix_ly8)
f.close()
f = h5py.File("features/ly10.h5", "w")
f.create_dataset("activations", data=matrix_ly10)
f.close()
f = h5py.File("features/ly12.h5", "w")
f.create_dataset("activations", data=matrix_ly12)
f.close()
f = h5py.File("features/ly14.h5", "w")
f.create_dataset("activations", data=matrix_ly14)
f.close()
f = h5py.File("features/ly16.h5", "w")
f.create_dataset("activations", data=matrix_ly16)
f.close()
