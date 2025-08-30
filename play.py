import torch
import torch.nn as nn
import numpy as np
from collections import deque

class DDPM:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def p_sample(self, model, x_t, t):
        """Standard DDPM sampling step"""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t]).reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1 / self.alphas[t]).reshape(-1, 1, 1, 1)
        
        # Model predicts noise
        predicted_noise = model(x_t, t)
        
        # Calculate mean
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(betas_t) * noise
        else:
            return model_mean

class PNDM:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, order=4):
        """
        PNDM uses higher-order numerical methods for better sampling
        order: Order of the numerical method (typically 2-4)
        """
        self.num_timesteps = num_timesteps
        self.order = order
        
        # Same noise schedule as DDPM
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # PNDM specific parameters
        self.noise_list = deque(maxlen=order)
        self.time_list = deque(maxlen=order)
        
        # Plms coefficients for different orders
        self.plms_coefficients = {
            2: [1.0, 1.0],
            3: [23/12, -16/12, 5/12],
            4: [55/24, -59/24, 37/24, -9/24]
        }
    
    def get_x0_prediction(self, model, x_t, t):
        """Predict x_0 from current state"""
        predicted_noise = model(x_t, t)
        alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        x0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        return x0_pred, predicted_noise

    def plms_sample(self, model, x_t, t):
        """
        Pseudo Linear Multistep sampling - the core innovation of PNDM
        Uses previous steps to make a more accurate prediction
        """
        # Get current prediction
        x0_pred, predicted_noise = self.get_x0_prediction(model, x_t, t)
        
        # Add to noise history
        self.noise_list.append(predicted_noise)
        self.time_list.append(t)
        
        # If we don't have enough history, fall back to regular sampling
        if len(self.noise_list) < self.order:
            return self.p_sample_ddpm(model, x_t, t)
        
        # Calculate PLMS estimate using historical values
        coefs = self.plms_coefficients[self.order]
        noise_estimate = sum(c * n for c, n in zip(coefs, self.noise_list))
        
        # Calculate parameters for the update
        alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, 1, 1, 1)
        alpha_cumprod_t_prev = self.alphas_cumprod[t-1].reshape(-1, 1, 1, 1) if t[0] > 0 else torch.ones_like(alpha_cumprod_t)
        
        # PLMS update step
        x_prev = (
            torch.sqrt(alpha_cumprod_t_prev) * x0_pred +
            torch.sqrt(1 - alpha_cumprod_t_prev) * noise_estimate
        )