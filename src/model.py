import torch

class Agent(torch.nn.Module):
	def __init__(self, side_length):
		super().__init__()

		self.squares = side_length*side_length
		self.fc1 = torch.nn.Linear(self.squares, 128)
		self.fc2 = torch.nn.Linear(128, 4)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.relu(self.fc1(x))
		x = self.fc2(x)

		return x