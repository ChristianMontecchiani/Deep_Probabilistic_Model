import torch
import torch.optim as optim

# Load dataset based on the 'dataset parameter'
def load_dataset(dataset: str, batch_size:int, transform):
	try:
		if dataset == "MNIST": 		
			trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
			trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

		elif dataset == "FashionMNIST":
			trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
			trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

		elif dataset == "EMNIST":
			trainset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
			trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

		return trainset, trainloader
	except: 
		print("Dataset is not in these values: [MNIST, FashionMNIST, EMNIST]")


# Define the training function
def train(model, train_loader, num_epochs, optimizer, learning_rate):
	# Choose the device where to train the model
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Chooe the optimizer to train the model 
	mean_loss = list()

	for epoch in range(num_epochs):
		for i, (data, _) in enumerate(train_loader):

			reconstructed, z_mean, z_log_var = model(inputs)
			loss = loss_function(reconstructed, inputs, z_mean, z_log_var) # Compute the loss 
			mean_loss.append(loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			if i % 10 == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))


# Define the loss function
def loss_function(reconstructed, inputs, z_mean, z_log_var):
    recon_loss = nn.functional.mse_loss(reconstructed, inputs, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return recon_loss + kl_div