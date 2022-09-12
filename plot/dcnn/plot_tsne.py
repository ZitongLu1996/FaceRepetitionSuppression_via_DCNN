from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt

def draw_tsne(features, imgs):
    print(f">>> t-SNE fitting")
    tsne = TSNE(n_components=2, init='pca', perplexity=40)
    Y = tsne.fit_transform(features)
    print("fitting over")

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 14.4)
    plt.axis('off')
    print("plotting images")
    imscatter(Y[:, 0], Y[:, 1], imgs, zoom=0.1, ax=ax)
    print("plot over")
    plt.savefig(fname='figure.jpg', dpi=600)
    plt.show()


def imscatter(x, y, images, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, image in zip(x, y, images):
        im = cv2.imread(image)
        im = cv2.resize(im, (320, 405))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_f = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(im_f, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

features = np.array(h5py.File("../../dcnn/features_random/ly16.h5", "r")["activations"])
print(features.shape)
imgs = []
conditions = ["f", "u", "s"]
for condition in conditions:
    for i in range(150):
        imgs.append("../../stimuli/withborder/" + condition + str(i+1).zfill(3) + ".jpg")

draw_tsne(features, imgs)