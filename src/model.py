import torch

class Agent(torch.nn.Module):
	def __init__(self, side_length):
		super().__init__()

		self.input_size = side_length*side_length*3

		self.fc1 = torch.nn.Linear(self.input_size, 2048)
		self.fc2 = torch.nn.Linear(2048, 512)
		self.fc3 = torch.nn.Linear(512, 4)
		# self.fc4 = torch.nn.Linear(64, 4)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = torch.relu(self.fc3(x))
		# x = torch.relu(self.fc4(x))

		return x