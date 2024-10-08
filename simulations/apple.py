import taichi as ti
import numpy as np
import torch
import utils
from engine.mpm_solver import MPMSolver
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in-dir', type=str, help='Input folder')
    parser.add_argument('-o', '--out-dir', type=str, help='Output folder')
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--offset', type=float, nargs='+', default=[0.0, 0.25])
    args = parser.parse_args()
    print(args)
    return args

args = parse_args()

def save_positions_pt(positions, iteration):
    positions = scaler.inverse(positions)
    positions_tensor = torch.from_numpy(positions).reshape(-1, 3, 2)
    filename = args.out_dir + f'/triangles/{iteration:04d}.pt'
    torch.save(positions_tensor, filename)

class Rescale:
    def __init__(self, scale=1.0, offset=[0.0, 0.0]):
        self.min = None
        self.max = None
        self.scale = scale
        self.offset = offset
    
    def fit(self, x):
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)
    
    def transform(self, x):
        return self.scale * (x - self.min) / (self.max - self.min) + self.offset
    
    def inverse(self, x):
        return (x - self.offset) / self.scale * (self.max - self.min) + self.min

write_to_disk = args.out_dir is not None
if write_to_disk:
    os.makedirs(f'{args.out_dir}/triangles', exist_ok=True)
    os.makedirs(f'{args.out_dir}/img', exist_ok=True)

# ti.init(arch=ti.cuda, device_memory_fraction=0.8)  # Try to run on GPU
ti.init(arch=ti.cpu, device_memory_GB=12)

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41, show_gui=False)

triangles = torch.load(f'{args.in_dir}/triangles.pt').cpu().numpy()
pts = triangles.reshape(triangles.shape[0] * 3, 3)
pts = pts[:, [0, 2]]

scaler = Rescale(args.scale, args.offset)
scaler.fit(pts)
pts = scaler.transform(pts)

mpm = MPMSolver(res=(64, 64), E_scale=9)

mpm.add_particles(particles=pts,
                  material=MPMSolver.material_elastic)

particles = mpm.particle_info()
save_positions_pt(particles['position'], 0)
for frame in range(1, 100):
    mpm.step(1e-2)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)
    particles = mpm.particle_info()
    save_positions_pt(particles['position'], frame)
    gui.circles(particles['position'],
                radius=1.5,
                color=colors[particles['material']])
    gui.show(f'{args.out_dir}/img/{frame:06d}.png' if write_to_disk else None)
