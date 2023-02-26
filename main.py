import os
import hydra
import time 
import numpy as np

import torch 
import torchvision.datasets as datasets
from torchvision import transforms
from torch.nn import MSELoss
from torchvision.utils import save_image
from torch.optim import Adam
from torch.utils.data import DataLoader

from trian import train
from model import DeepProbabilisticModel

import warnings


# Ignora gli avvisi specifici
warnings.filterwarnings("ignore")

	
		

# # Funzione per salvare l'immagine 
# def save(model, dataloader, name, save_dir):
# 	# Salvataggio delle immagini ricostruite
# 	image = iter(dataloader).next()[0]
# 	image = image.view(image.size(0), -1)
# 	output = model(image)
# 	#output = output.view(28, 28)

# 	print("Saving images")
# 	image = np.array(image.detach().cpu() * 255, dtype=np.uint8).reshape(28,28)
# 	output = np.array(output.detach().cpu() * 255, dtype=np.uint8).reshape(28,28)

# 	saving_file_name = save_dir + f'{name}.png'
# 	saving_file_name_out = save_dir + f'{name}_reconstruded.png'

# 	# Save the output to the disk
# 	image = Image.fromarray(image)
# 	output = Image.fromarray(output)

# 	image.save(saving_file_name)
# 	output.save(saving_file_name_out)

@hydra.main(config_path='cfg', config_name='config')
def main(cfg):
	
	# Variabili da definire
	batch_size = cfg.batch_size
	learning_rate = cfg.learning_rate
	num_epochs = cfg.num_epochs
	dataset = cfg.dataset
	save_path = cfg.save_path + f"compressed_model_{dataset}.pth"
	
	
	
	# Definisci trasformazioni per le immagini
	transform = transforms.Compose([
        transforms.ToTensor(),  
    ])
	
	# Read Dataset 
	trainset, trainloader, testloader = load_dataset(dataset, batch_size, transform)
	latent_dim, input_dim, output_dim = 20, 784, 784

	# Create model and optimizer
	model = DeepProbabilisticModel(latent_dim, input_dim, output_dim)
	optimizer = Adam(model.parameters(), lr= learning_rate)

	# Train the model
	if cfg.train:
		train(model, train_loader, num_epochs, optimizer, learning_rate)

	else:
		model = DeepProbabilisticModel()
		model.load_state_dict(torch.load(save_path))
		save(model, testloader, name, cfg.save_dir)

if __name__ == '__main__':
	main()