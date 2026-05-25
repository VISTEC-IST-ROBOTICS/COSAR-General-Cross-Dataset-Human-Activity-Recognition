# data/preprocessing/pgba.py
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from tqdm import tqdm
from termcolor import colored
from copy import deepcopy
import numpy as np

class PGBA:
    def __init__(self, b=50, h=0.1, verbose=False):
        self.b = b
        self.h = h
        self.verbose = verbose
        
        self.gravity_templates = {
            'standing': torch.tensor([0.0, 0.0, -1.0]),
            'lying': torch.tensor([1.0, 0.0, 0.0])
        }
    
    def __call__(self, dataset):
        return self.compute(dataset)

    def compute(self, dataset):
        BAR_DESC_WIDTH  = 30
        n_subjects      = len(dataset['data'])
        imu_positions   = dataset['imu_positions']
        classes         = dataset['classes']
        
        DEBUG_PROFILE = False

        if DEBUG_PROFILE:
            ctx = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                with_stack=True,
            )
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx as prof:
            scores = {'standing': [], 'lying': []}
            
            for s in tqdm(range(n_subjects), desc=f'PGBA: b={self.b},h={self.h}'.ljust(BAR_DESC_WIDTH), unit='subject'):
                if self.verbose: print(colored(f'Realigning subject {s+1}/{n_subjects}', 'green'))
                x = dataset['data'][s]
                
                each_scores = {'standing': [], 'lying': []}
                
                for sensor_idx, sensor_pos in enumerate(imu_positions):
                    if x['standing'].ndim == 3:
                        dimension = (sensor_idx)
                    else:
                        dimension = (slice(None), sensor_idx)

                    x_s = x['standing'][dimension].float().clone()
                    x_l = x['lying'][dimension].float().clone()
                    
                    v_s = self.find_KDE_peak(x_s, b=self.b, h=self.h)
                    v_l = self.find_KDE_peak(x_l, b=self.b, h=self.h)
                    
                    v_s = v_s / torch.norm(v_s)
                    v_l = v_l / torch.norm(v_l)

                    with (record_function("find_rotation_matrix") if DEBUG_PROFILE else nullcontext()):
                        optimal_R = self._pgba(v_s, v_l)
                        
                    if self.verbose:
                        v_rot_s = self._multiply_matrices(optimal_R, v_s)
                        v_rot_l = self._multiply_matrices(optimal_R, v_l)
                        
                        self.verbose = False
                        err_1, err_2 = self.euclidean_error(v_rot_s, v_rot_l)
                        self.verbose = True
                        
                        each_scores['standing'].append(err_1.item())
                        each_scores['lying'].append(err_2.item())
                        print('-------------------------------------')
                
                    with (record_function("apply_rotations") if DEBUG_PROFILE else nullcontext()):
                        for activity in classes:
                            if dataset['data'][s][activity] is None:
                                continue
                            
                            raw     = deepcopy(x[activity][dimension].float())
                            # raw = x[activity][dimension].float().clone()
                            rotated = self._multiply_matrices(optimal_R, raw)
                            dataset['data'][s][activity][dimension] = rotated
                            
                if self.verbose:
                    all_score_standing = torch.mean(torch.tensor(each_scores['standing']))
                    all_score_lying    = torch.mean(torch.tensor(each_scores['lying']))

                    txt = f"All score of lying={all_score_lying:.4f}, standing={all_score_standing:.4f}\n"
                    if any(x > 0.35 for x in [all_score_lying, all_score_standing]):
                        print(colored(txt, 'red'))
                    else:
                        print(txt)
                    scores['standing'].append(all_score_standing)
                    scores['lying'].append(all_score_lying)
                    
            if self.verbose:
                mean = []
                for activity in ['standing', 'lying']:
                    all_scores = torch.stack(scores[activity])
                    mean.append(torch.mean(all_scores))
                print(colored(f'[Summary] Final scores across all subjects: standing={mean[0]}, lying={mean[1]}', 'green'))

        if DEBUG_PROFILE and prof is not None:
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        return dataset
    
    def K_function(self, x):
        return (1 / torch.sqrt(torch.tensor(2.0 * torch.pi))) * torch.exp(-0.5 * x**2)

    def find_KDE_peak(self, x, b=50, h=0.1):
        """
        Apply KDE peak finding over time dimension
        Args:
            x: [n_ch, times] or [bs, n_ch, times]
        Returns: 
            [n_ch] or [bs, n_ch]
        """
        # Handle batch dimension
        if x.ndim == 3:
            return torch.stack([self.find_KDE_peak(x[i], b, h) for i in range(x.size(0))], dim=0)  # [bs, n_ch]
        
        # Original 2D case [n_ch, times]
        n_axes = x.size(0)
        peaks  = []
        
        for i in range(n_axes):
            x_i = x[i]  # [times]
            
            min_val, max_val = x_i.min(), x_i.max()
            eval_points = torch.linspace(min_val, max_val, b)

            diff          = (eval_points[None, :] - x_i[:, None]) / h  # [times, num_points]
            kernel_values = self.K_function(diff)                        # [times, num_points]
            density       = kernel_values.sum(dim=0) / (x_i.size(0) * h)  # [num_points]
            
            peak_idx = density.argmax()
            peaks.append(eval_points[peak_idx])

        return torch.stack(peaks)  # [n_ch]
    
    def _pgba(self, v_s, v_l):    
        """
        Args:
            v_s, v_l: [3] or [bs, 3]
        Returns:
            optimal_R: [3, 3] or [bs, 3, 3]
        """
        # Handle batch dimension — process each sample, then stack
        if v_s.ndim == 2:
            return torch.stack([self._pgba(v_s[i], v_l[i]) for i in range(v_s.size(0))], dim=0)  # [bs, 3, 3]
        
        optimal_R = torch.eye(3)
        
        # ======= First rotation: align standing =======
        if self.verbose: print('└── Rotation 1: Align standing')
    
        R = self.RMC(v_s, self.gravity_templates['standing'])
        
        # Accomulate rotation
        optimal_R = R @ optimal_R
        
        # Rotate signal to update all vectors for next iteration
        v1 = self._multiply_matrices(optimal_R, v_s)
        v2 = self._multiply_matrices(optimal_R, v_l)
        
        if self.verbose: 
            self._print_vector(v1, v2)
            err_1, err_2 = self.euclidean_error(v1, v2)
            
        # ======= Second rotation: align lying =======
        if self.verbose: print('└── Rotation 2: Align lying')
        R = self.RMC(v2, self.gravity_templates['lying'], k_fix=torch.tensor([0.0, 0.0, -1.0]))
        
        # Accomulate rotation
        optimal_R = R @ optimal_R
        
        # Rotate signal to update all vectors for next iteration
        v1 = self._multiply_matrices(optimal_R, v_s)
        v2 = self._multiply_matrices(optimal_R, v_l)
        
        if self.verbose: 
            self._print_vector(v1, v2)
            err_1, err_2 = self.euclidean_error(v1, v2)
        
        return optimal_R

    def RMC(self, v, v_ref, k_fix=None):
        """
        Find the rotation matrix using Rodrigues' rotation formula
        Args:
            v: [3] - a unit vector
            v_ref: [3] - the reference unit vector we want to align to
        """
        np.set_printoptions(precision=4, suppress=True)
        threshold   = 0.96
        mapping     = {0:(1,2), 1:(0,2), 2:(0,1)}
        
        # Get the cosine similarity between the two vectors
        c = torch.dot(v, v_ref)
        if c > threshold:
            if self.verbose:
                print(colored('    ├── Already aligned, no rotation needed', 'yellow'))
            return torch.eye(3).float()
        elif c < -threshold:
            if k_fix is not None:
                k = k_fix
            else:
                rotation_axis = torch.argmin(v)
                k = torch.nn.functional.one_hot(rotation_axis, num_classes=3)
            angle = torch.tensor(np.pi)  # 180 degrees in radians
            if self.verbose:
                print(colored(f'    ├── Opposite, rotating 180 degrees with {k}', 'yellow'))
        else:
            if k_fix is not None: 
                k = k_fix
            else:
                k_fix = torch.linalg.cross(v, v_ref)
                max_idx = torch.abs(k_fix).argmax()
                k = torch.nn.functional.one_hot(max_idx, num_classes=len(k_fix)).float()
                k = k * torch.sign(k_fix[max_idx])
                # k = k_fix
        
            cross_prod = torch.linalg.cross(v, v_ref)
            sin_theta = torch.dot(cross_prod, k) / (torch.norm(v) * torch.norm(v_ref))
            cos_theta = torch.dot(v, v_ref) / (torch.norm(v) * torch.norm(v_ref))
            
            angle   = torch.atan2(sin_theta, cos_theta)  # Now returns [-π, π]
            deg     = torch.sign(angle) * torch.rad2deg(torch.abs(angle))
            if self.verbose:
                print(f'        ├── Angle to rotate: {deg:.2f} with axis {k_fix.cpu().numpy()} -> {k.cpu().numpy()}')
        
        V = [
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ]
        V = torch.tensor(V).float()
        I = torch.eye(3)
        
        R = I + torch.sin(angle) * V + ((1 - torch.cos(angle)) * (V @ V))
        return R
        
    
    def rotation_matrix(self, degrees, dim):
        """Your existing rotation matrix code with small optimization"""
        theta = torch.deg2rad(torch.tensor(degrees))
        # theta = np.radians(degree)
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(theta), -torch.sin(theta)],
            [0, torch.sin(theta), torch.cos(theta)] 
        ])
        Ry = torch.tensor([
            [torch.cos(theta), 0, torch.sin(theta)],
            [0, 1, 0],
            [-torch.sin(theta), 0, torch.cos(theta)]
        ])
        Rz = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ])
        no_rotation = torch.eye(3)
        mapping = {(0,1): Rz, (0,2): Ry, (1,2): Rx, (0,0): no_rotation}
        return torch.inverse(mapping[dim])  # Inverse to rotate back
    
    def _multiply_matrices(self, R, v):
        if R.ndim == 2:
            # Single rotation matrix [3, 3]
            if v.ndim == 1:
                return R @ v                                # [3]
            elif v.ndim == 2:
                return R @ v                                # [3, t]
            else:
                return torch.einsum('ij, bjt->bit', R, v)  # [b, 3, t]
        else:
            # Batched rotation matrix [bs, 3, 3]
            if v.ndim == 2:
                return torch.einsum('bij, jt->bit', R, v)  # [bs, 3, t]
            else:
                return torch.einsum('bij, bjt->bit', R, v) # [bs, 3, t]
        
    def _print_vector(self, v1, v2):
        np.set_printoptions(precision=4, suppress=True)
        print(f'        ├── standing    {v1.cpu().numpy()}')
        print(f'        ├── lying       {v2.cpu().numpy()}')
    
    def euclidean_error(self, v1, v2):
        ref_1 = self.gravity_templates['standing']
        ref_2 = self.gravity_templates['lying']

        dist_1 = torch.norm(v1 - ref_1)
        dist_2 = torch.norm(v2 - ref_2)

        if self.verbose:
            threshold = 0.35
            msg = f'        └── Euclidean error after two rotations: standing={dist_1:.4f}, lying={dist_2:.4f}'
            print(colored(msg, 'red') if dist_1 > threshold or dist_2 > threshold else msg)

        return dist_1, dist_2
    
    def angular_error(self, v1, v2):
        ref_1 = self.gravity_templates['standing']
        ref_2 = self.gravity_templates['lying']

        # Standing error
        cross_1   = torch.linalg.cross(v1, ref_1)
        k_1       = cross_1 / torch.norm(cross_1)           # rotation axis (unit vector)
        sin_1     = torch.dot(cross_1, k_1)                 # norm=1, so sin = |cross|
        cos_1     = torch.dot(v1, ref_1)                    # norm=1, so cos = dot
        angle_1   = torch.atan2(sin_1, cos_1) * (180.0 / torch.pi)

        # Lying error
        cross_2   = torch.linalg.cross(v2, ref_2)
        k_2       = cross_2 / torch.norm(cross_2)
        sin_2     = torch.dot(cross_2, k_2)
        cos_2     = torch.dot(v2, ref_2)
        angle_2   = torch.atan2(sin_2, cos_2) * (180.0 / torch.pi)

        if self.verbose:
            threshold = 10.0
            msg = f'        └── Angular error after two rotations: standing={angle_1:.4f}°, lying={angle_2:.4f}°'
            print(colored(msg, 'red') if angle_1 > threshold or angle_2 > threshold else msg)

        return angle_1, angle_2
    
    # cross_prod = torch.linalg.cross(v, v_ref)
        # sin_theta = torch.dot(cross_prod, k) / (torch.norm(v) * torch.norm(v_ref))
        # cos_theta = torch.dot(v, v_ref) / (torch.norm(v) * torch.norm(v_ref))
        
        # angle   = torch.atan2(sin_theta, cos_theta)  # Now returns [-π, π]