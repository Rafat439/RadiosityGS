import torch
from torch import nn
from dnnlib import EasyDict
from submodules.env_map_sampler import *
from utils.sh_utils import RGB2SH, SH2RGB

class LightModel:
    def __init__(self, dataset):
        self._intensity = None
        self.optimizer = EasyDict(step=lambda *args, **kwargs: True, zero_grad=lambda *args, **kwargs: True, load_state_dict=lambda *args, **kwargs: True, state_dict=lambda *args, **kwargs: {})
        
        self.is_directional_light = True
        self.max_sh_degree = dataset.sh_degree
    
    @property
    def get_scaling(self):
        return torch.tensor([1e-3, 1e-3], device="cuda")[None].repeat(len(self._xyz), 1)
    
    @property
    def get_rotation(self):
        return torch.tensor([1., 0., 0., 0.], device="cuda")[None].repeat(len(self._xyz), 1)
    
    @property
    def get_geovalue(self):
        return torch.tensor([6.], device="cuda")[None].repeat(len(self._xyz), 1)
    
    @property
    def get_norm_factor(self):
        return torch.tensor([1.], device="cuda")[None].repeat(len(self._xyz), 1)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_is_light_source(self):
        return torch.tensor([True], device="cuda")[None].repeat(len(self._xyz), 1)
    
    @property
    def get_emissions(self):
        return torch.cat((torch.relu(self._intensity), torch.zeros_like(self._intensity).repeat(1, (self.max_sh_degree + 1) ** 2 - 1, 1)), dim=1)
    
    @property
    def get_env_map(self):
        pred_env_map = torch.zeros(3, 16, 32).cuda()
        scattered_pred_env_map = SH2RGB(self.get_emissions[:, 0, :])
        for i in range(512):
            pred_env_map[:, i // 32, i % 32] = scattered_pred_env_map[i]
        return pred_env_map
    
    def save(self, path: str):
        torch.save({
            "intensity": self._intensity, 
            "optimizer": self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        data = torch.load(path)
        self._intensity = data["intensity"]
        self.optimizer.load_state_dict(data["optimizer"])
    
    def create_from_env_map(self, env_map = None, init_intensity = None):
        if env_map is None:
            env_map = torch.ones((3, 128, 256), dtype=torch.float, device="cuda")
            sample_dict = sample_env_map(env_map, num_samples=16, radius=1000)

            self._xyz = sample_dict['means3D']
            self._intensity = nn.Parameter((torch.ones_like(sample_dict['emissions'][:, None, :]) * init_intensity).detach().clone().requires_grad_(True))
        else:
            env_map = torch.nn.functional.interpolate(env_map[None], (16, 32), mode='bilinear', align_corners=False)[0]
            sample_dict = sample_env_map(env_map, num_samples=16, radius=1000)

            self._xyz = sample_dict['means3D']
            self._intensity = nn.Parameter(RGB2SH(sample_dict['emissions'][:, None, :]).detach().clone().requires_grad_(True))
    
    def training_setup(self, training_args):
        l = [
            {'params': [self._intensity], 'lr': training_args.env_map_lr, "name": "intensity"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    def from_camera_if_possible(self, camera):
        if camera.pl_pos is not None and camera.pl_intensity is not None:
            return EasyDict(
                get_xyz=camera.pl_pos[None], 
                get_geovalue=torch.tensor([6.])[None].to(camera.data_device), 
                get_norm_factor=torch.tensor([1.])[None].to(camera.data_device), 
                get_scaling=torch.tensor([1e-3, 1e-3]).float().to(camera.data_device)[None], 
                get_rotation=torch.tensor([1., 0., 0., 0.]).float().to(camera.data_device)[None], 
                get_emissions=torch.cat((camera.pl_intensity, torch.zeros_like(camera.pl_intensity).repeat((self.max_sh_degree + 1) ** 2 - 1, 1)))[None], 
                get_is_light_source=torch.tensor([True])[None].to(camera.data_device), 
                is_directional_light=False
            )
        else:
            return self