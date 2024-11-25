import torch
from ldm_patched.k_diffusion.sampling import BrownianTreeNoiseSampler
from ldm_patched.modules.model_sampling import EPS, ModelSamplingDiscrete


class ModelSampling(EPS, ModelSamplingDiscrete):
    pass

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

    
def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

class KSampler:

    def __init__(self,
                 latent, 
                 steps,
                 device, 
                 sampler='dpmpp_2m_sde_gpu',
                 scheduler='karras',
                 denoise=1, 
                 model_options={},
                 start_step=0, 
                 last_step=30,
                 force_full_denoise=False, 
                 seed = None):
        self.device = device
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise  # denoising_strength
        self.model_options = model_options
        
        # step param
        self.old_denoised = None
        self.h_last = None
        
        self.model_sampling = ModelSampling()
        

        sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            assert start_step < (len(sigmas) - 1)
            sigmas = sigmas[start_step:]
        
            # if start_step < (len(sigmas) - 1):
            #     sigmas = sigmas[start_step:]
            # else:
            #     if latent_image is not None:
            #         return latent_image
            #     else:
            #         return torch.zeros_like(noise)
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        self.noise_sampler = BrownianTreeNoiseSampler(latent, sigma_min, sigma_max, seed=seed)
        self.sigmas = sigmas
        self.log_sigmas = sigmas.log()

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in ['dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2']:
            steps += 1
            discard_penultimate_sigma = True
            
        sigmas = get_sigmas_karras(n=steps, sigma_min=0.0292, sigma_max=14.6146)
        # sigmas = get_sigmas_karras(n=steps, sigma_min=0.0291675, sigma_max=14.614642)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas
    
    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            new_steps = int(steps/denoise)
            sigmas = self.calculate_sigmas(new_steps).to(self.device)
            self.sigmas = sigmas[-(steps + 1):]

    @torch.no_grad()
    def step(self, i, pred_x0, x, t=None, eta=1., s_noise=1., solver_type='midpoint'):
        """DPM-Solver++(2M) SDE."""

        if solver_type not in {'heun', 'midpoint'}:
            raise ValueError('solver_type must be \'heun\' or \'midpoint\'')
        sigmas = self.sigmas

        denoised = pred_x0
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if self.old_denoised is not None:
                r = self.h_last / h
                if solver_type == 'heun':
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - self.old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - self.old_denoised)

            if eta:
                x = x + self.noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

            self.old_denoised = denoised
            self.h_last = h
        return x

    def timestep(self, i):
        sigma = self.sigmas[i]
        t = self.model_sampling.timestep(sigma).float()
        return t

    def calculate_input(self, i, x):
        sigma = self.sigmas[i]
        return self.model_sampling.calculate_input(sigma, x)

    def calculate_denoised(self, i, model_output, model_input):
        sigma = self.sigmas[i]
        return self.model_sampling.calculate_denoised(sigma, model_output, model_input)


