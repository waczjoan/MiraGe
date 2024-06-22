import torch
import os
import trimesh
from os import makedirs



def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in mesh_f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', filepath)



p = torch.load('pseudomesh.pt')
faces = torch.range(0, p.shape[0] * 3 - 1).reshape(p.shape[0],3)
vertice = p.reshape(p.shape[0] * 3, 3)

filename = f'jump_one_gs_flat2d_rots_set_1_lr_rot_not_0_size_tr_20.obj'
write_simple_obj(mesh_v=vertice.detach().cpu().numpy(), mesh_f=faces, filepath=filename)

