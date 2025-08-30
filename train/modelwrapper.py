import torch
import pytorch_lightning as pl

class SanzioModule(pl.LightningModule):
    def __init__(self, pipeline , unet_model, sampler, ddim_steps, lr):
        super().__init__()
        self.pipeline = pipeline
        self.lr = lr
        self.unet_model = unet_model
        self.sampler = sampler
        self.ddim_steps = ddim_steps
        
        self.prefix = 'Raffaello Sanzio Painting'
    
    def training_step(self, batch, batch_idx):
        ims, titles = batch

        text = [f"Raffaello Sanzio Painting, {title}" for title in titles]
        input_text_emb = self.pipeline.get_learned_conditioning(text)
        
        latents = self.pipeline.get_first_stage_encoding(self.pipeline.encode_first_stage(ims))
        latents.requires_grad_(True)  

        noise = torch.randn_like(latents)
        
        t = torch.randint(0, self.ddim_steps, (latents.shape[0],), device=self.device, dtype=torch.long)
        
        noisy_latents = self.sampler.stochastic_encode(latents, t, noise=noise)
        
        noise_pred = self.pipeline.apply_model(noisy_latents, t, input_text_emb)
        noise_pred.requires_grad_(True) 
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.unet_model.parameters(), lr=self.lr)
        return optimizer