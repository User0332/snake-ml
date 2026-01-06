import torch
import gameserializer

Games = list[gameserializer.Game]

def flatten_board(board: list[list[float]]) -> list[float]:
	return [cell for row in board for cell in row]

def snake_body_length(turn: gameserializer.Turn) -> int:
	return flatten_board(turn.board).count(0.33)

def get_score(game: gameserializer.Game) -> int:
	return snake_body_length(game[-1])-2

def filter_by_score(games: Games, min_score: int) -> Games:
	return [game for game in games if get_score(game) >= min_score]

def truncate_game_at_last_fruit(game: gameserializer.Game) -> gameserializer.Game:
	for i, frame in enumerate(game):
		if snake_body_length(frame) == get_score(game) + 2:
			return game[:i]
		
	return game

def truncate_all_at_last_fruit(games: Games) -> Games:
	return [truncate_game_at_last_fruit(game) for game in games]

def map_board_to_next_move(game: gameserializer.Game) -> gameserializer.MappedFlattenedGame:
	mapped_game: gameserializer.MappedFlattenedGame = []

	for i in range(len(game)-1):
		mapped_game.append(
			gameserializer.MappedFlattenedTurn(
				next_move=game[i+1].direction,
				curr_board=flatten_board(game[i].board)
			)
		)

	return mapped_game

def map_all_to_next_move(games: Games) -> list[gameserializer.MappedFlattenedGame]:
	return [map_board_to_next_move(game) for game in games]

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
		board=[list(row) for row in zip(*turn.board[::-1])]
	)

	return rotated

def rotate_game_to_all_directions(game: gameserializer.Game) -> Games:
	rotated_games: Games = [game]

	last_rot = game

	for _ in range(3):
		rotated_game: gameserializer.Game = []

		for turn in last_rot:
			rotated_game.append(rotate_turn_90_degrees(turn))

		last_rot = rotated_game

		rotated_games.append(rotated_game)

	return rotated_games

def rotate_all_games_to_all_directions(games: Games) -> Games:
	all_rotated_games: Games = []

	for game in games:
		all_rotated_games.extend(rotate_game_to_all_directions(game))

	return all_rotated_games

games = gameserializer.read_games()


games = filter_by_score(games, min_score=8)
games = truncate_all_at_last_fruit(games)
games = rotate_all_games_to_all_directions(games)


mapped_games = map_all_to_next_move(games)

DIRECTION_LABEL_MAP = {
	"UP": 0,
	"RIGHT": 1,
	"DOWN": 2,
	"LEFT": 3
}

REVERSE_DIRECTION_LABEL_MAP = {v: k for k, v in DIRECTION_LABEL_MAP.items()}

def conv_direction_to_y_tensor(direction: str) -> list[int]:
	directions = [0, 0, 0, 0]

	directions[DIRECTION_LABEL_MAP[direction]] = 1

	return directions

x_tensor = torch.tensor([
	turn.curr_board for game in mapped_games for turn in game
], dtype=torch.float32)

y_tensor = torch.tensor([
	conv_direction_to_y_tensor(turn.next_move) for game in mapped_games for turn in game
], dtype=torch.float32)