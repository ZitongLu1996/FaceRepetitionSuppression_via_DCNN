import numpy as np
import h5py

sim_f = np.zeros([8, 75*149])
sim_u = np.zeros([8, 75*149])
sim_s = np.zeros([8, 75*149])

for ly in range(8):
    rdm = np.array(h5py.File("rdms_afterPCA/ly" + str((ly+1)*2) + ".h5", "r")["rdm"])

    n = 75*149
    index = 0
    sim = np.zeros([75*149])

    for i in range(150):
        for j in range(150):
            if i < j:
                sim_f[ly, index] = 1-rdm[i, j]
                sim_u[ly, index] = 1-rdm[i+150, j+150]
                sim_s[ly, index] = 1-rdm[i+300, j+300]
                index = index + 1

f = h5py.File("similarities/familiar.h5", "w")
f.create_dataset("similarities", data=sim_f)
f.close()

f = h5py.File("similarities/unfamiliar.h5", "w")
f.create_dataset("similarities", data=sim_u)
f.close()

f = h5py.File("similarities/scrambled.h5", "w")
f.create_dataset("similarities", data=sim_s)
f.close()

sim_f = np.zeros([8, 75*149])
sim_u = np.zeros([8, 75*149])
sim_s = np.zeros([8, 75*149])

for ly in range(8):
    rdm = np.array(h5py.File("rdms_random_afterPCA/ly" + str((ly+1)*2) + ".h5", "r")["rdm"])

    n = 75*149
    index = 0
    sim = np.zeros([75*149])

    for i in range(150):
        for j in range(150):
            if i < j:
                sim_f[ly, index] = 1-rdm[i, j]
                sim_u[ly, index] = 1-rdm[i+150, j+150]
                sim_s[ly, index] = 1-rdm[i+300, j+300]
                index = index + 1

f = h5py.File("similarities/random_familiar.h5", "w")
f.create_dataset("similarities", data=sim_f)
f.close()

f = h5py.File("similarities/random_unfamiliar.h5", "w")
f.create_dataset("similarities", data=sim_u)
f.close()

f = h5py.File("similarities/random_scrambled.h5", "w")
f.create_dataset("similarities", data=sim_s)
f.close()