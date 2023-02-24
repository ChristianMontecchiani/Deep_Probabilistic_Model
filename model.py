import torch
import torch.nn as nn


class DeepProbabilisticModel(nn.Module):

	def __init(self, latent_dim, input_dim, output_dim):
		super(DeepProbabilisticModel, self).__init__()

		self.endoder_layers = nn.Sequential(
			nn.Linear(input_dim, 512),
			nn.ReLU(), 
			nn.Linear(512, 256), 
			nn.ReLU(), 
			nn.Linear(256, latent_dim + latent_dim),
		)

		self.decoder_layers = nn.Sequential(
			nn.Linear(latent_dim, 256),
			nn.ReLU(), 
			nn.Linear(256, 512),
			nn.ReLU(), 
			nn.Linear(512, output_dim),
		)
	
	def sample_latent_space(self, z_mean, z_log_var):
		eps = torch.radn_like(z_mean)
		return eps * torch.exp(0.5 * z_log_var) + z_mean
	
	def forward(self, x):
        # Encode the input into mean and variance of the latent variables
        z_mean_log_var = self.encoder_layers(x)
        z_mean = z_mean_log_var[:, :latent_dim]
        z_log_var = z_mean_log_var[:, latent_dim:]

        # Sample the latent variables from the mean and variance
        z = self.sample_latent_variables(z_mean, z_log_var)

        # Decode the latent variables into the output
        reconstructed = self.decoder_layers(z)

        # Return the reconstructed output and the mean and variance of the latent variables
        return reconstructed, z_mean, z_log_var