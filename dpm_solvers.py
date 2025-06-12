# DPM solvers: stiff semi-linear ODE
# Note: hyperparams of Atom_SDE and Bond_SDE should keep the same for DPM-Solver-1, DPM-Solver-2 and DPM-Solver-3 !!!

import torch
import numpy as np
import functools
import torch.nn.functional as F
from models.utils import get_multi_theta_fn, get_multi_score_fn, get_theta_fn
import os 
# from . import datasets

#     data.property = [ 'gasa', 'qed', 'sascore', 'fsp3', 'mce-18', 'npscore', 'alarm_nmr-rule',    'bms-rule', 'chelating-rule', 'pains', 'lipinski-rule', 'pfizer-rule', 
#     'gsk-rule', 'goldentriangle', 'colloidal-aggregators', 'fluc-inhibitors', 
#     'blue-fluorescence', 'green_fluorescence', 'reactive-compounds',
#     'caco-2-permeability', 'mdck-Permeability', 'pampa', 'pgp-inhibitor', 
#     'pgp-substrate', 'hia', 'f20', 'f30', 'f50',
#     'ppb', 'vdss', 'fu', 'bbb', 'oatp1b1-inhibitor', 'oatp1b3-inhibitor', 
#     'bcrp-inhibitor', 'mrp1-inhibitor'
# ]

TARGET_PROPERTIES = [
     0, 0.51, 3.344, 0.2, 60, 2.304, 2, 1, 1, 1, 0, 0, 0, 0, 0.657, 0.596, 
     0.118, 0.037, 0.466, -6.176708723, -4.932220949, 0.588069022, 0.000759878, 
     0.083957948, 0.001433551, 0.798006043, 0.971844275, 0.99640698, 93.01566438, 
     0.027556217, 11.8208783, 0.101334056, 0.8847965, 0.810298979, 0.009897998, 
     0.776696146
 ]

# TARGET_PROPERTIES = [
#    1, 0.821, 6.738, 0.357, 68.421, 0.528, 0, 0, 0, 0, 0, 0, 0, 0, 0.559, 
#    0.596, 0.359, 0.481, 0.012, -5.395283552, -4.915619939, 0.08498501033, 
#    0.07419760525, 0.6957918406, 0.00522005558, 0.00505053997, 0.01434534788, 
#    0.2680845261, 31.49749045, 0.4108096321, 63.94362374, 0.6977955103, 
#    0.9463194609, 0.9726055264, 0.2094644755, 0.7872437239
# ]


def sample_nodes(n_nodes_pmf, atom_shape, device):
    n_nodes = torch.multinomial(n_nodes_pmf, atom_shape[0], replacement=True)
    atom_mask = torch.zeros((atom_shape[0], atom_shape[1]), device=device)
    for i in range(atom_shape[0]):
        atom_mask[i][:n_nodes[i]] = 1.
    bond_mask = (atom_mask[:, None, :] * atom_mask[:, :, None]).unsqueeze(1)
    bond_mask = torch.tril(bond_mask, -1)
    bond_mask = bond_mask + bond_mask.transpose(-1, -2)
    return n_nodes, atom_mask, bond_mask


def expand_dim(x, n_dim):
    if n_dim == 3:
        x = x[:, None, None]
    elif n_dim == 4:
        x = x[:, None, None, None]
    return x


def dpm1_update(x_last, t_last, t_i, sde, theta):
    # dpm_solver 1 order update function
    expand_fn = functools.partial(expand_dim, n_dim=len(x_last.shape))

    lambda_i, alpha_i, std_i = sde.log_snr(t_i)
    lambda_last, alpha_last, _ = sde.log_snr(t_last)
    h_i = lambda_i - lambda_last

    x_i = expand_fn(alpha_i / alpha_last) * x_last - expand_fn(std_i * torch.expm1(h_i)) * theta
    return x_i




def dpm_mol_solver_3(atom_sde, bond_sde, theta_fn, x_atom_last, x_bond_last,
                     t_last, t_i, atom_mask, bond_mask, r1=1./3., r2=2./3. , *args,**kwargs):
    vec_t_last = torch.ones(x_atom_last.shape[0], device=x_atom_last.device) * t_last
    vec_t_i = torch.ones(x_atom_last.shape[0], device=x_atom_last.device) * t_i
    atom_fn = functools.partial(expand_dim, n_dim=len(x_atom_last.shape))
    bond_fn = functools.partial(expand_dim, n_dim=len(x_bond_last.shape))

    lambda_i, alpha_i, std_i = atom_sde.log_snr(vec_t_i)
    lambda_last, alpha_last, _ = atom_sde.log_snr(vec_t_last)
    h_i = lambda_i - lambda_last

    s1 = atom_sde.lambda2t(lambda_last + r1 * h_i)
    s2 = atom_sde.lambda2t(lambda_last + r2 * h_i)

    _, alpha_s1, std_s1 = atom_sde.log_snr(s1)
    _, alpha_s2, std_s2 = atom_sde.log_snr(s2)

    properties = kwargs['properties']
# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
    # --- Property-Guided Score Calculation ---
    def get_guided_scores(x_atom, x_bond, t):
        # Enable gradients only for the guidance step
        with torch.enable_grad():
            # Create new tensors that require gradients
            x_atom_grad = x_atom.detach().requires_grad_(True)
            x_bond_grad = x_bond.detach().requires_grad_(True)
            
            # Forward pass with gradient-enabled inputs
            atom_theta, bond_theta, pred_props = theta_fn(
                (x_atom_grad, x_bond_grad), t,
                atom_mask=atom_mask, bond_mask=bond_mask,
                properties=properties
            )
            
            # Calculate property loss
            prop_loss = F.mse_loss(pred_props, properties, reduction='none').mean()

            # Compute gradients
            grad_atom, grad_bond = torch.autograd.grad(
                outputs=prop_loss,
                inputs=[x_atom_grad, x_bond_grad],
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )

        guidance_scale = 0.5
        # Return original outputs with applied guidance
        return (
            atom_theta - guidance_scale * grad_atom.detach() ,
            bond_theta - guidance_scale * grad_bond.detach() 
        )
    
    atom_theta_0, bond_theta_0 = get_guided_scores(x_atom_last, x_bond_last, vec_t_last)
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends

    # atom_theta_0, bond_theta_0 = theta_fn((x_atom_last, x_bond_last), vec_t_last,
    #                                       atom_mask=atom_mask, bond_mask=bond_mask)

    tmp_lin = alpha_s1 / alpha_last
    tmp_nonlin = std_s1 * torch.expm1(r1 * h_i)
    u_atom_1 = atom_fn(tmp_lin) * x_atom_last - atom_fn(tmp_nonlin) * atom_theta_0
    u_bond_1 = bond_fn(tmp_lin) * x_bond_last - bond_fn(tmp_nonlin) * bond_theta_0

# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here      
    atom_theta_s1, bond_theta_s1 = get_guided_scores(u_atom_1, u_bond_1, s1)
        
    # atom_theta_s1, bond_theta_s1 , pred_props = theta_fn((u_atom_1, u_bond_1), s1, atom_mask=atom_mask, bond_mask=bond_mask,properties = properties)
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends

    # atom_theta_s1, bond_theta_s1 = theta_fn((u_atom_1, u_bond_1), s1, atom_mask=atom_mask, bond_mask=bond_mask)


    D_atom_1 = atom_theta_s1 - atom_theta_0
    D_bond_1 = bond_theta_s1 - bond_theta_0

    tmp_lin = alpha_s2 / alpha_last
    tmp_nonlin1 = std_s2 * torch.expm1(r2 * h_i)
    tmp_nonlin2 = (std_s2 * r2 / r1) * (torch.expm1(r2 * h_i) / (r2 * h_i) - 1)
    u_atom_2 = atom_fn(tmp_lin) * x_atom_last - atom_fn(tmp_nonlin1) * atom_theta_0 - atom_fn(tmp_nonlin2) * D_atom_1
    u_bond_2 = bond_fn(tmp_lin) * x_bond_last - bond_fn(tmp_nonlin1) * bond_theta_0 - bond_fn(tmp_nonlin2) * D_bond_1

# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
    atom_theta_s2, bond_theta_s2 = get_guided_scores(u_atom_2, u_bond_2, s2)
    # atom_theta_s2, bond_theta_s2 , pred_props = theta_fn((u_atom_2, u_bond_2), s2, atom_mask=atom_mask, bond_mask=bond_mask,properties = properties)
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends


    # atom_theta_s2, bond_theta_s2 = theta_fn((u_atom_2, u_bond_2), s2, atom_mask=atom_mask, bond_mask=bond_mask)

    D_atom_2 = atom_theta_s2 - atom_theta_0
    D_bond_2 = bond_theta_s2 - bond_theta_0

    tmp_lin = alpha_i / alpha_last
    tmp_nonlin1 = std_i * torch.expm1(h_i)
    tmp_nonlin2 = (std_i / r2) * (torch.expm1(h_i) / h_i - 1)
    x_atom_i = atom_fn(tmp_lin) * x_atom_last - atom_fn(tmp_nonlin1) * atom_theta_0 - atom_fn(tmp_nonlin2) * D_atom_2
    x_bond_i = bond_fn(tmp_lin) * x_bond_last - bond_fn(tmp_nonlin1) * bond_theta_0 - bond_fn(tmp_nonlin2) * D_bond_2

    return x_atom_i, x_bond_i




def get_mol_sampler_dpm3(atom_sde, bond_sde, atom_shape, bond_shape, inverse_scaler,
                         time_step, eps=1e-3, denoise=False, device='cuda',config = None):
    # arrange time schedule
    num_step = int(time_step // 3)

    def sampler(model, n_nodes_pmf=None, time_point=None, z=None, atom_mask=None, bond_mask=None, theta_fn=None,*args, **kwargs):
        
# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here   

        # target_properties = kwargs.pop( 'target_properties', getattr(config.eval, 'property_targets', None))

        
        # Load normalization stats
        processed_dir = '/workspace/CDGS-NEW/data/zinc250k_7props/processed'
        stats_path = os.path.join(processed_dir, 'prop_stats.pt')
        prop_stats = torch.load(stats_path)
        prop_mean = prop_stats['mean'].numpy()
        prop_std = prop_stats['std'].numpy()

        # Normalize target properties
        target_properties = (TARGET_PROPERTIES - prop_mean) / prop_std


        # target_properties = TARGET_PROPERTIES

        if target_properties is not None:
            # Convert to tensor with batch dimension
            properties_tensor = torch.tensor(
                np.array(target_properties, dtype=np.float32),
                device=device
            ).float()
            
            # Add batch dimension and expand to match atom_shape[0] (batch size)
            batch_size = atom_shape[0]  # Get batch size from input shape
            properties = properties_tensor.unsqueeze(0).expand(batch_size, -1)
            


# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends

        if time_point is None:
            start_lambda = atom_sde.log_snr_np(atom_sde.T)
            stop_lambda = atom_sde.log_snr_np(eps)
            lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
            time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]
        else:
            start_time, stop_time = time_point
            start_lambda = atom_sde.log_snr_np(start_time)
            stop_lambda = atom_sde.log_snr_np(stop_time)
            lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
            time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]

        with torch.no_grad():
            # set up dpm theta func
            if theta_fn is None:
# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
                theta_fn = get_multi_theta_fn(atom_sde, bond_sde, model, properties = properties,train=False, continuous=True,*args, **kwargs)
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends

            else:
                theta_fn = theta_fn

            # initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distribution of the SDE.
                x_atom = atom_sde.prior_sampling(atom_shape).to(device)
                x_bond = bond_sde.prior_sampling(bond_shape).to(device)

                # Sample the number of nodes, if z is None
                n_nodes, atom_mask, bond_mask = sample_nodes(n_nodes_pmf, atom_shape, device)
                x_atom = x_atom * atom_mask.unsqueeze(-1)
                x_bond = x_bond * bond_mask
            else:
                # just use the concurrent prior z and node_mask, bond_mask
                x_atom, x_bond = z
                n_nodes = atom_mask.sum(-1).long()

            # run solver func according to time schedule
            t_last = time_steps[0]
            for t_i in time_steps[1:]:
# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
                x_atom, x_bond  = dpm_mol_solver_3(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                  atom_mask, bond_mask,properties = properties)
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends

                t_last = t_i

            if denoise:
                pass

            x_atom = inverse_scaler(x_atom, atom=True) * atom_mask.unsqueeze(-1)
            x_bond = inverse_scaler(x_bond, atom=False) * bond_mask
            return x_atom, x_bond, num_step * 3, n_nodes

    return sampler


def get_mol_encoder_dpm3(atom_sde, bond_sde, time_step, eps=1e-3, device='cuda'):
    # arrange time schedule
    num_step = int(time_step // 3)

    def sampler(model, batch, time_point=None,*args,**kwargs):

# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
        # target_properties = kwargs.pop( 'target_properties', getattr(config.eval, 'property_targets', None))

        target_properties = TARGET_PROPERTIES
    # Convert to tensor
        if target_properties is not None:
            kwargs['target_properties'] = torch.tensor(
            np.array(target_properties, dtype=np.float32),
            device=device).float()

# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends

        if time_point is None:
            start_lambda = atom_sde.log_snr_np(atom_sde.T)
            stop_lambda = atom_sde.log_snr_np(eps)
            lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
            time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]
            time_steps.reverse()
        else:
            start_time, stop_time = time_point
            start_lambda = atom_sde.log_snr_np(start_time)
            stop_lambda = atom_sde.log_snr_np(stop_time)
            lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
            time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]

        with torch.no_grad():
            # set up dpm theta func
# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
            theta_fn = get_multi_theta_fn(atom_sde, bond_sde, model, train=False, continuous=True,*args,**kwargs)
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends
            # theta_fn = get_multi_theta_fn(atom_sde, bond_sde, model, train=False, continuous=True)

            # run forward deterministic diffusion process
            x_atom, atom_mask, x_bond, bond_mask = batch

            # run solver func according to time schedule
            t_last = time_steps[0]
            for t_i in time_steps[1:]:
# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
                x_atom, x_bond = dpm_mol_solver_3(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                  atom_mask, bond_mask,*args,**kwargs)
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends
                # pdb.set_trace()
                t_last = t_i

            return x_atom, x_bond, num_step * 3

    return sampler


def get_mol_sampler_dpm_mix(atom_sde, bond_sde, atom_shape, bond_shape, inverse_scaler,
                            time_step, eps=1e-3, denoise=False, device='cuda'):
    # arrange time schedule
    num_step = int(time_step // 3)

    start_lambda = atom_sde.log_snr_np(atom_sde.T)
    stop_lambda = atom_sde.log_snr_np(eps)
    lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step+1)
    time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]

    R = int(time_step) % 3
    # time_steps = np.linspace(start=atom_sde.T, stop=eps, num=num_step + 1)

    def sampler(model, n_nodes_pmf, z=None):
        with torch.no_grad():
            # set up dpm theta func
            theta_fn = get_multi_theta_fn(atom_sde, bond_sde, model, train=False, continuous=True)

            # initial sample
            assert z is None
            # If not represent, sample the latent code from the prior distribution of the SDE.
            x_atom = atom_sde.prior_sampling(atom_shape).to(device)
            x_bond = bond_sde.prior_sampling(bond_shape).to(device)

            # Sample the number of nodes, if z is None
            n_nodes, atom_mask, bond_mask = sample_nodes(n_nodes_pmf, atom_shape, device)
            x_atom = x_atom * atom_mask.unsqueeze(-1)
            x_bond = x_bond * bond_mask

            # run solver func according to time schedule
            t_last = time_steps[0]

            if R == 0:
                for t_i in time_steps[1:-2]:
                    x_atom, x_bond = dpm_mol_solver_3(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                      atom_mask, bond_mask)
                    t_last = t_i
                t_i = time_steps[-2]
                x_atom, x_bond = dpm_mol_solver_2(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                  atom_mask, bond_mask)
                t_last = t_i
                t_i = time_steps[-1]
                x_atom, x_bond = dpm_mol_solver_1(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                  atom_mask, bond_mask)
            else:
                for t_i in time_steps[1:-1]:
                    x_atom, x_bond = dpm_mol_solver_3(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                      atom_mask, bond_mask)
                    t_last = t_i
                t_i = time_steps[-1]
                if R == 1:
                    x_atom, x_bond = dpm_mol_solver_1(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                      atom_mask, bond_mask)
                elif R == 2:
                    x_atom, x_bond = dpm_mol_solver_2(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                                                      atom_mask, bond_mask)
                else:
                    raise ValueError('Step Error in mix DPM-solver.')

            if denoise:
                pass

            x_atom = inverse_scaler(x_atom, atom=True) * atom_mask.unsqueeze(-1)
            x_bond = inverse_scaler(x_bond, atom=False) * bond_mask
            return x_atom, x_bond, time_step, n_nodes

    return sampler


def get_sampler_dpm3(sde, shape, inverse_scaler, time_step, eps=1e-3, denoise=False, device='cuda'):
    # arrange time schedule
    num_step = int(time_step // 3)

    def sampler(model, n_nodes_pmf=None, time_point=None, z=None, mask=None, theta_fn=None,*args, **kwargs):

# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
        # target_properties = kwargs.pop( 'target_properties', getattr(config.eval, 'property_targets', None))

        target_properties = TARGET_PROPERTIES
    # Convert to tensor
        if target_properties is not None:
            kwargs['target_properties'] = torch.tensor(
            np.array(target_properties, dtype=np.float32),
            device=device).float()

# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends

        if time_point is None:
            start_lambda = sde.log_snr_np(sde.T)
            stop_lambda = sde.log_snr_np(eps)
            lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
            time_steps = [sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]
        else:
            start_time, stop_time = time_point
            start_lambda = sde.log_snr_np(start_time)
            stop_lambda = sde.log_snr_np(stop_time)
            lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
            time_steps = [sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]

        with torch.no_grad():
            # set up dpm theta func

            if theta_fn is None:
# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
                 theta_fn = get_theta_fn(sde, model, train=False, continuous=True,*args,**kwargs)
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends
            
                # theta_fn = get_theta_fn(sde, model, train=False, continuous=True)
            else:
                theta_fn = theta_fn

            # initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distribution of the SDE.
                x = sde.prior_sampling(shape).to(device)
                # Sample the number of nodes, if z is None
                n_nodes = torch.multinomial(n_nodes_pmf, shape[0], replacement=True)
                mask = torch.zeros((shape[0], shape[-1]), device=device)
                for i in range(shape[0]):
                    mask[i][:n_nodes[i]] = 1.
                mask = (mask[:, None, :] * mask[:, :, None]).unsqueeze(1)

            else:
                x = z
                batch_size, _, max_num_nodes, _ = mask.shape
                node_mask = mask[:, 0, 0, :].clone()  # without checking correctness
                node_mask[:, 0] = 1
                n_nodes = node_mask.sum(-1).long()

            # run solver func according to time schedule
            t_last = time_steps[0]
            for t_i in time_steps[1:]:

# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
                x = dpm_solver_3(sde, theta_fn, x, t_last, t_i, mask,*args,**kwargs)
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends

                t_last = t_i

            if denoise:
                pass

            x = inverse_scaler(x) * mask
            return x, num_step * 3, n_nodes

    return sampler


def get_mol_dpm3_twostage(atom_sde, bond_sde, atom_shape, bond_shape, inverse_scaler,
                          time_step, eps=1e-3, denoise=False, device='cuda'):
    # arrange time schedule
    num_step = int(time_step // 3)

    def sampler(model, n_nodes_pmf, time_point, guided_theta_fn,properties = None,*args,**kwargs):

# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
        # target_properties = kwargs.pop( 'target_properties', getattr(config.eval, 'property_targets', None))

        target_properties = TARGET_PROPERTIES
    # Convert to tensor
        if target_properties is not None:
            kwargs['target_properties'] = torch.tensor(
            np.array(target_properties, dtype=np.float32),
            device=device).float()

# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends

        start_lambda = atom_sde.log_snr_np(atom_sde.T)
        stop_lambda = atom_sde.log_snr_np(eps)
        lambda_sched = np.linspace(start=start_lambda, stop=stop_lambda, num=num_step + 1)
        time_steps = [atom_sde.lambda2t_np(lambda_ori) for lambda_ori in lambda_sched]

        with torch.no_grad():
            # set up dpm theta func
# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
            theta_fn = get_multi_theta_fn(atom_sde, bond_sde, model, train=False, continuous=True,*args,**kwargs)
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends
            # theta_fn = get_multi_theta_fn(atom_sde, bond_sde, model, train=False, continuous=True)

            # initial sample
            x_atom = atom_sde.prior_sampling(atom_shape).to(device)
            x_bond = bond_sde.prior_sampling(bond_shape).to(device)

            # Sample the number of nodes, if z is None
            n_nodes, atom_mask, bond_mask = sample_nodes(n_nodes_pmf, atom_shape, device)
            x_atom = x_atom * atom_mask.unsqueeze(-1)
            x_bond = x_bond * bond_mask

            # run solver func according to time schedule
            t_last = time_steps[0]
            for t_i in time_steps[1:]:
                if t_last > time_point:
# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
                    x_atom, x_bond = dpm_mol_solver_3(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i, atom_mask, bond_mask,*args,**kwargs)
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends
                    # x_atom, x_bond = dpm_mol_solver_3(atom_sde, bond_sde, theta_fn, x_atom, x_bond, t_last, t_i,
                    #                                   atom_mask, bond_mask)
                else:
# ---------------------------------------------------------------------------------------------------------------------------------#                    
# Change here             
                    x_atom, x_bond = dpm_mol_solver_3(atom_sde, bond_sde, guided_theta_fn, x_atom, x_bond, t_last, t_i,atom_mask, bond_mask,*args,**kwargs)
# ---------------------------------------------------------------------------------------------------------------------------------#        
# Change ends
                    # x_atom, x_bond = dpm_mol_solver_3(atom_sde, bond_sde, guided_theta_fn, x_atom, x_bond, t_last, t_i,
                    #                                   atom_mask, bond_mask)
                t_last = t_i

            if denoise:
                pass

            x_atom = inverse_scaler(x_atom, atom=True) * atom_mask.unsqueeze(-1)
            x_bond = inverse_scaler(x_bond, atom=False) * bond_mask
            return x_atom, x_bond, num_step * 3, n_nodes

    return sampler
