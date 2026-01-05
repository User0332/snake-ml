from dataclasses import dataclass
import dill
from pygame import Vector2

Game = list['Turn']

@dataclass
class Turn:
	direction: str
	board: list[int]

def serialize_game(game: Game) -> None:
	try: data: list['Game'] = dill.load(open('games.dill', 'rb'))
	except FileNotFoundError: data = []

	data.append(game)

	dill.dump(data, open('games.dill', 'wb'))


def serialize_frame(food_pos: tuple[int, int], snake_pos: list[tuple[int, int]], board_size: tuple[int, int], direction: Vector2) -> Turn:
	board = [[0 for _ in range(board_size[1])] for _ in range(board_size[0])]

	board[snake_pos[0][0]][snake_pos[0][1]] = 0.67

	for pos in snake_pos[1:]:
		board[pos[0]][pos[1]] = 0.33
	
	board[food_pos[0]][food_pos[1]] = 1

	return Turn(
		direction=conv_direction(direction),
		board=[cell for row in board for cell in row] # flatten the board into a vector
	)

def conv_direction(direction: Vector2) -> int:
	if direction == Vector2(1, 0):
		return "RIGHT"
	elif direction == Vector2(-1, 0):
		return "LEFT"
	elif direction == Vector2(0, 1):
		return "DOWN"
	elif direction == Vector2(0, -1):
		return "UP"
	else:
		return "NONE"