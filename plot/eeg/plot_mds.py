import h5py
import numpy as np
from sklearn.manifold import MDS
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import matplotlib.pyplot as plt

rdms = np.array(h5py.File("../../eeg/acc-ctrdms/avg_tbt-acc-rdms.h5", "r")["rdm"])

rdm = rdms[37]

mds = MDS(n_components=2, dissimilarity="precomputed")
embedding = mds.fit(rdm)
pos = embedding.fit_transform(rdm)
print(pos)

def imscatter(x, y, images, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, image in zip(x, y, images):
        im = cv2.imread(image)
        im = cv2.resize(im, (1920, 2430))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_f = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(im_f, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

imgs = ["demo_stimuli/fn.jpg", "demo_stimuli/fe.jpg", "demo_stimuli/fl.jpg",
        "demo_stimuli/un.jpg", "demo_stimuli/ue.jpg", "demo_stimuli/ul.jpg",
        "demo_stimuli/sn.jpg", "demo_stimuli/se.jpg", "demo_stimuli/sl.jpg"]
img_path = "demo_stimuli/"

fig, ax = plt.subplots()
fig.set_size_inches(21.6, 14.4)
plt.axis('off')
imscatter(pos[:, 0], pos[:, 1], imgs, zoom=0.1, ax=ax)
plt.xlim(-0.7, 0.7)
plt.ylim(-0.7, 0.7)
#plt.savefig(fname='figure.eps', format='eps')
plt.title("250 ms", fontsize=50)
plt.savefig("rdm_mds/250ms.jpg", dpi=600)
plt.show()
