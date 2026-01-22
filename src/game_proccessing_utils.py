from dataclasses import dataclass

import numpy as np
import gameserializer

@dataclass
class SplitFrames:
	# early - mid - late game models data distribution
	early_frames: list[gameserializer.MappedFlattenedTurn]
	mid_frames: list[gameserializer.MappedFlattenedTurn]
	late_frames: list[gameserializer.MappedFlattenedTurn]

def flatten_board(board: list[list[float]]) -> list[float]:
	return [cell for row in board for cell in row]

def snake_body_length(turn: gameserializer.Turn) -> int:
	return flatten_board(turn.board).count(0.33)

def get_score(game: gameserializer.Game) -> int:
	return snake_body_length(game[-1])-2

def filter_by_score(games: gameserializer.Games, min_score: int) -> gameserializer.Games:
	return [game for game in games if get_score(game) >= min_score]

# remove losing frames from data
def truncate_game_at_last_fruit(game: gameserializer.Game) -> gameserializer.Game:
	for i, frame in enumerate(game):
		if snake_body_length(frame) == get_score(game) + 1:
			return game[:i]
		
	return game

def truncate_all_at_last_fruit(games: gameserializer.Games) -> gameserializer.Games:
	return [truncate_game_at_last_fruit(game) for game in games]

# create three boards to differentiate objects in grid
def create_separate_boards(board: list[float]) -> tuple[list[float], list[float], list[float]]:
	head_board = [0]*len(board)
	body_board = head_board.copy()
	fruit_board = head_board.copy()

	for i in range(len(board)):
		if board[i] == 0.33: body_board[i] = 1
		elif board[i] == 0.67: head_board[i] = 1
		elif board[i] == 1: fruit_board[i] = 1

	return head_board, body_board, fruit_board

def map_board_to_next_move(game: gameserializer.Game) -> gameserializer.MappedFlattenedGame:
	mapped_game: gameserializer.MappedFlattenedGame = []

	for i in range(len(game)-1):
		flattened_board = flatten_board(game[i].board)

		head_board, body_board, fruit_board = create_separate_boards(flattened_board)

		mapped_game.append(
			gameserializer.MappedFlattenedTurn(
				next_move=game[i+1].direction,
				curr_direction=game[i].direction,
				head_board=head_board,
				body_board=body_board,
				fruit_board=fruit_board,
				combined_board=flattened_board
			)
		)

	return mapped_game

def map_all_to_next_move(games: gameserializer.Games) -> gameserializer.MappedFlattenedGames:
	return [map_board_to_next_move(game) for game in games]

# methods to rotate boards to different directions to provide more training data; unused because they bloat the dataset and reduce performance
def rotate_direction_90_degrees(direction: str) -> str:
	if direction == "UP":
		return "RIGHT"
	elif direction == "RIGHT":
		return "DOWN"
	elif direction == "DOWN":
		return "LEFT"
	elif direction == "LEFT":
		return "UP"
	
	raise ValueError(f"Invalid direction: {direction}")


def rotate_turn_90_degrees(turn: gameserializer.Turn) -> gameserializer.Turn:
	rotated = gameserializer.Turn(
		direction=rotate_direction_90_degrees(turn.direction),
		board=[list(row) for row in np.rot90(turn.board, k=-1)]
	)

	return rotated

def rotate_game_to_all_directions(game: gameserializer.Game) -> gameserializer.Games:
	rotated_games: gameserializer.Games = [game]

	last_rot = game

	for _ in range(3):
		rotated_game: gameserializer.Game = []

		for turn in last_rot:
			rotated_game.append(rotate_turn_90_degrees(turn))

		last_rot = rotated_game

		rotated_games.append(rotated_game)

	return rotated_games

def rotate_all_games_to_all_directions(games: gameserializer.Games) -> gameserializer.Games:
	all_rotated_games: gameserializer.Games = []

	for game in games:
		all_rotated_games.extend(rotate_game_to_all_directions(game))

	return all_rotated_games

DIRECTION_LABEL_MAP = {
	"UP": 0,
	"RIGHT": 1,
	"DOWN": 2,
	"LEFT": 3
}

REVERSE_DIRECTION_LABEL_MAP = {v: k for k, v in DIRECTION_LABEL_MAP.items()}

OPPOSITE_DIRECTION_MAP = {
	DIRECTION_LABEL_MAP["UP"]: DIRECTION_LABEL_MAP["DOWN"],
	DIRECTION_LABEL_MAP["DOWN"]: DIRECTION_LABEL_MAP["UP"],
	DIRECTION_LABEL_MAP["RIGHT"]: DIRECTION_LABEL_MAP["LEFT"],
	DIRECTION_LABEL_MAP["LEFT"]: DIRECTION_LABEL_MAP["RIGHT"]
}

# converts the direction label to an expected/result tensor
def conv_direction_to_y_tensor(direction: str, curr_direction: str) -> list[int]:
	directions = [0, 0, 0, 0]

	directions[DIRECTION_LABEL_MAP[direction]] = 1

	return directions

def split_frames_by_length(games: gameserializer.MappedFlattenedGames, early_thresh: int, mid_thresh: int) -> SplitFrames:
	early_frames = []
	mid_frames = []
	late_frames = []

	for game in games:
		for frame in game:
			body_length = frame.body_board.count(1)
			
			if body_length <= early_thresh:
				early_frames.append(frame)
				continue

			if body_length <= mid_thresh:
				mid_frames.append(frame)
				continue

			late_frames.append(frame)


	return SplitFrames(early_frames, mid_frames, late_frames)