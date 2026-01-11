from dataclasses import dataclass
import dill
from pygame import Vector2

Game = list['Turn']
Games = list[Game]
MappedFlattenedGame = list['MappedFlattenedTurn']
MappedFlattenedGames = list[MappedFlattenedGame]

@dataclass
class MappedFlattenedTurn:
	next_move: str
	curr_direction: str
	head_board: list[float]
	body_board: list[float]
	fruit_board: list[float]
	combined_board: list[float]

@dataclass
class Turn:
	direction: str
	board: list[list[float]]

def read_games(fname: str='games.dill') -> list[Game]:
	try:
		data: list[Game] = dill.load(open(fname, 'rb'))
	except FileNotFoundError:
		data = []
		
	return data

def serialize_game(game: Game, fname: str='games.dill') -> None:
	if game[-1].direction == "NONE": return # invalid game, do not serialize
	
	try: data: list['Game'] = dill.load(open(fname, 'rb'))
	except FileNotFoundError: data = []

	data.append(game)

	dill.dump(data, open(fname, 'wb'))


def serialize_frame(food_pos: tuple[int, int], snake_pos: list[tuple[int, int]], board_size: tuple[int, int], direction: Vector2) -> Turn:
	board = [[0 for _ in range(board_size[1])] for _ in range(board_size[0])]

	try:
		board[snake_pos[0][0]][snake_pos[0][1]] = 0.67
	except IndexError: # snake head is out of board
		pass

	for pos in snake_pos[1:]:
		board[pos[0]][pos[1]] = 0.33
	
	board[food_pos[0]][food_pos[1]] = 1

	return Turn(
		direction=conv_direction(direction),
		board=board
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