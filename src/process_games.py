import torch
from game_proccessing_utils import conv_direction_to_y_tensor, filter_by_score, map_all_to_next_move, rotate_all_games_to_all_directions, split_frames_by_length, truncate_all_at_last_fruit
import gameserializer
from processing_constants import EARLY_LENGTH_THRESH, MID_LENGTH_THRESH, MIN_SCORE_THRESH, USE_MULTIPLE_MODELS, USE_REFEED_TRAINING

games = gameserializer.read_games()

if USE_REFEED_TRAINING: games+=gameserializer.read_games('refeed-games.dill')

games = filter_by_score(games, min_score=MIN_SCORE_THRESH)
games = truncate_all_at_last_fruit(games)
# games = rotate_all_games_to_all_directions(games)

mapped_games = map_all_to_next_move(games)
split_frames = split_frames_by_length(mapped_games, early_thresh=EARLY_LENGTH_THRESH, mid_thresh=MID_LENGTH_THRESH)

print(f"Mapped Games: {len(mapped_games)}")
print(f"Total frames: {len([turn for game in mapped_games for turn in game])}")

def conv_frames_to_tensors(frames: list[gameserializer.MappedFlattenedTurn]) -> tuple[torch.Tensor, torch.Tensor]:
	x_tensor = torch.tensor([
		turn.head_board+turn.body_board+turn.fruit_board for turn in frames
	], dtype=torch.float32)

	y_tensor = torch.tensor([
		conv_direction_to_y_tensor(turn.next_move, turn.curr_direction) for turn in frames
	], dtype=torch.float32)

	return x_tensor, y_tensor

if USE_MULTIPLE_MODELS:
	print(f"Early frames: {len(split_frames.early_frames)}")
	print(f"Mid frames: {len(split_frames.mid_frames)}")
	print(f"Late frames: {len(split_frames.late_frames)}")

if USE_MULTIPLE_MODELS:
	early_x_tensor, early_y_tensor = conv_frames_to_tensors(split_frames.early_frames)
	mid_x_tensor, mid_y_tensor = conv_frames_to_tensors(split_frames.mid_frames)
	late_x_tensor, late_y_tensor = conv_frames_to_tensors(split_frames.late_frames)
else:
	early_x_tensor, early_y_tensor = conv_frames_to_tensors([frame for game in mapped_games for frame in game])
	mid_x_tensor = mid_y_tensor = late_x_tensor = late_y_tensor = None