import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import os
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pickle

torch.manual_seed(0)
np.random.seed(0)

import SimPEG.data_misfit as data_misfit

from SimPEG.electromagnetics.static import resistivity as dc

from discretize import TreeMesh, TensorMesh

from SimPEG.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    apparent_resistivity_from_voltage,
    plot_pseudosection,
)

from SimPEG import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
    utils,
)

from SimPEG.utils import sdiag

from SimPEG import SolverLU as Solver

from discretize.utils import mkvc, refine_tree_xyz, active_from_xyz

from SimPEG.utils.io_utils.io_utils_electromagnetics import read_dcip2d_ubc

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import copy
from time import time

class NFTOMO(nn.Module):
    def __init__(self):
         super(NFTOMO, self).__init__()
         self.b1 = nn.Sequential(nn.Linear(2,128), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b2 = nn.Sequential(nn.Linear(128,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b3 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b4 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b5 = nn.Sequential(nn.Linear(256,256), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b6 = nn.Sequential(nn.Linear(256,128), nn.LeakyReLU(inplace = True, negative_slope=0.2))
         self.b7 = nn.Sequential(nn.Linear(128,1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.b1(x)

        x2 = self.b2(x1)

        x3 = self.b3(x2)

        x4 = self.b4(x3)

        x5 = self.b5(x4)

        x6 = self.b6(x5)

        x7 = self.b7(x6)


        return torch.mul(x7,-8)

torch.manual_seed(0)
np.random.seed(0)
model = NFTOMO()

x1 = np.linspace(-1,1,45)
x2 = np.linspace(-1,1,200)
input_list = []
for i in x1:
  for k in x2:
    input_list.append([i,k])
input_ = torch.tensor(input_list, dtype=torch.float32)

epochs = 2000
lr_ = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr_)
# Cell sizes
csx, csy = 5.0, 5.0
# Number of core cells in each direction
ncx, ncy = 200.0 , 45.0
# Number of padding cells to add in each direction
npad = 7
# Vectors of cell lengths in each direction with padding
hx_ = [(csx, npad, -1.5), (csx, ncx), (csx, npad, 1.5)]
hy_ = [(csy, npad, -1.5), (csy, ncy)]
# Create mesh and center it
mesh = TensorMesh([hx_, hy_], x0="CN")

# build ind_active
_ind_active = np.ones((45,200))
_ind_active = np.pad(_ind_active, ((7,0),(7,7)),  constant_values=(0, 0))
ind_active_=[]
for i in _ind_active.flatten():
  if i == 1.:
    ind_active_.append(True)
  else:
    ind_active_.append(False)
ind_active_=np.array(ind_active_)

# files to work with
dir_path = "./Case_3.3_truncated_z_225"
topo_filename = dir_path + "/topo_xyz.txt"
data_filename = dir_path + "/dc_data.obs"

topo_xyz = np.loadtxt(str(topo_filename))
dc_data = read_dcip2d_ubc(data_filename, "volt", "general")
dc_data.standard_deviation = 0.05 * np.abs(dc_data.dobs)

topo_2d = np.unique(topo_xyz[:, [0, 2]], axis=0)
ind_active = active_from_xyz(mesh, topo_2d)
survey = dc_data.survey
survey.drape_electrodes_on_topography(mesh, ind_active, option="top")

background_conductivity = np.log(1e-2)
active_map = maps.InjectActiveCells(mesh, ind_active_, np.exp(background_conductivity))

# Define mapping from model to active cells
nC = int(ind_active_.sum()) # Number of cells below the surface< ind_active.shape[0]
conductivity_map = active_map * maps.ExpMap()
simulation = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=survey, sigmaMap=conductivity_map, solver=Solver, storeJ=True
)

W = sdiag(1 / (dc_data.standard_deviation))

def theta_m(output):
    theta_m = torch.abs(torch.sub(output.flatten(), torch.ones(output.flatten().size(dim=0)), alpha = -4.6))
    return torch.sum(theta_m)

def total_loss(output, J, beta):
    """
    Note beta here differs from the beta in the convential EM inversion where beta typically starts with a value greater than 1 (e.x. 10), and then cooling down with a fixed rate.
    beta: [0,1]
    total_loss = (1-beta)*theta_phi + beta*theta_m
    """
    theta_phi =(1/(output.size(dim=0)))*torch.matmul(torch.from_numpy(J).type(torch.float32),output.flatten())
    theta_m_ = (1/(output.size(dim=0)))*theta_m(output)
    return (1-beta)*theta_phi + beta*theta_m_ 


diff_min = 1e16

def decay(x):
  return np.exp(-x/800)
diff_list = []
J_list = []
loss_list = []
beta_list = []
reg_list = []
m_list = []
directory_name = 'NF_3_3'
import os
os.mkdir(directory_name)

for epoch in range(epochs):
    beta = decay(epoch)
    output = model(input_)
    m = output.detach().flatten().numpy()
    print(np.max(m), np.min(m))
    simulation = dc.simulation_2d.Simulation2DNodal(mesh, survey=survey, sigmaMap=conductivity_map, solver=Solver, storeJ=True)
    J = simulation.Jtvec(m, W.T*(W*simulation.residual(m, dc_data.dobs)))
    loss = total_loss(output, J, beta)
    reg = theta_m(output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    diff = W*simulation.residual(m, dc_data.dobs)
    diff = 0.5*np.vdot(diff,diff)
    if epoch%100 == 0:
        pkl_name = '/test_'+str(epoch)+'.pkl'
        f = open(directory_name+pkl_name, 'wb')
        pickle.dump(m, f)
        pickle.dump(diff_list,f)
        pickle.dump(J_list,f)
        f.close()
    print("The ",epoch," epoch diff is ", diff)
    print("The ",epoch," epoch total loss is ", loss)
    print("The ",epoch," epoch beta is ", beta)
    print("The ",epoch," epoch reg is ", reg)
    diff_list.append(diff)
    J_list.append(J)
    loss_list.append(loss.detach().numpy())
    beta_list.append(beta)
    reg_list.append(reg.detach().numpy())
    m_list.append(m)


pkl_name = directory_name+'/test_final'+'.pkl'
f = open(pkl_name, 'wb')
pickle.dump(m, f)
pickle.dump(diff_list,f)
pickle.dump(J_list,f)
pickle.dump(loss_list,f)
pickle.dump(beta_list,f)
pickle.dump(reg_list,f)
pickle.dump(m_list,f)
f.close()

torch.save(model.state_dict(), directory_name+'/weights.pt')
torch.save(model, directory_name+'/model.pt')

#Return the final result
model_= NFTOMO()
model_.load_state_dict(torch.load(directory_name+'/weights.pt'))
output = model_(input_)
m = output.detach().numpy()[0][0].flatten()
pkl_name = directory_name+'/final'+'.pkl'
f = open(pkl_name, 'wb')
pickle.dump(m, f)
f.close()   
