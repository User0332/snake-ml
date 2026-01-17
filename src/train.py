#Imports
from model import Agent
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from process_games import (
	early_x_tensor, early_y_tensor,
	mid_x_tensor, mid_y_tensor,
	late_x_tensor, late_y_tensor
)
from processing_constants import USE_MULTIPLE_MODELS

def train_model(x_tensor: torch.Tensor, y_tensor: torch.Tensor, save_name: str, epochs: int, checkpoint_model: str=None):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	print(f"Training model {save_name} with device: {device}")

	x_tensor = x_tensor.to(device)
	y_tensor = y_tensor.to(device)

	x_train, x_test, y_train, y_test = train_test_split(
		x_tensor, y_tensor, test_size=0.2, random_state=42
	)

	if checkpoint_model:
		model = torch.load(checkpoint_model)
	else:
		model = Agent(side_length=10)

	model = model.to(device)

	loss_fn = nn.MSELoss()
	optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)

	EPOCHS = epochs
	losses = []
	for epoch in range(EPOCHS):
		model.train()
		optimizer_adam.zero_grad()

		outputs = model(x_train)
		loss = loss_fn(outputs, y_train)

		losses.append(loss.item())
		loss.backward()

		optimizer_adam.step()

		if (epoch + 1) % 250 == 0:
			print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

	model.eval()
	correct = 0
	total = y_test.shape[0]

	with torch.no_grad():
		outputs = model(x_test)
		predicted = torch.argmax(outputs, dim=1)
		actual = torch.argmax(y_test, dim=1)
		correct = (predicted == actual).sum().item()

	accuracy = correct / total * 100
	print(f"Test Accuracy: {accuracy:.2f}%")

	torch.save(model.to("cpu"), save_name)
	return model, accuracy

if USE_MULTIPLE_MODELS:
	train_model(early_x_tensor, early_y_tensor, "model_early.pt", 5000)
	train_model(mid_x_tensor, mid_y_tensor, "model_mid.pt", 5000)
	train_model(late_x_tensor, late_y_tensor, "model_late.pt", 5000)
else:
	train_model(early_x_tensor, early_y_tensor, "model_early.pt", 5000)