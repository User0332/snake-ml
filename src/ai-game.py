# !LICENSEINFO the contents of this file were originally created by https://github.com/clear-code-projects (unlicensed) with later modifications from Carl Furtado & Aidan McClure (MIT License)

import time
import pygame,sys,random
import torch
from pygame.math import Vector2

import game_proccessing_utils
import gameserializer
from processing_constants import COLLECT_REFEED_TRAINING_DATA, EARLY_LENGTH_THRESH, MID_LENGTH_THRESH, USE_MULTIPLE_MODELS, USE_REFEED_TRAINING

class SNAKE:
	def __init__(self):
		self.body = [Vector2(5,5),Vector2(4,5),Vector2(3,5)]
		self._direction = Vector2(0,0)
		self.last_direction = Vector2(0,0)
		self.new_block = False

		self.head_up = pygame.image.load('Graphics/head_up.png').convert_alpha()
		self.head_down = pygame.image.load('Graphics/head_down.png').convert_alpha()
		self.head_right = pygame.image.load('Graphics/head_right.png').convert_alpha()
		self.head_left = pygame.image.load('Graphics/head_left.png').convert_alpha()
		
		self.tail_up = pygame.image.load('Graphics/tail_up.png').convert_alpha()
		self.tail_down = pygame.image.load('Graphics/tail_down.png').convert_alpha()
		self.tail_right = pygame.image.load('Graphics/tail_right.png').convert_alpha()
		self.tail_left = pygame.image.load('Graphics/tail_left.png').convert_alpha()

		self.body_vertical = pygame.image.load('Graphics/body_vertical.png').convert_alpha()
		self.body_horizontal = pygame.image.load('Graphics/body_horizontal.png').convert_alpha()

		self.body_tr = pygame.image.load('Graphics/body_tr.png').convert_alpha()
		self.body_tl = pygame.image.load('Graphics/body_tl.png').convert_alpha()
		self.body_br = pygame.image.load('Graphics/body_br.png').convert_alpha()
		self.body_bl = pygame.image.load('Graphics/body_bl.png').convert_alpha()
		self.crunch_sound = pygame.mixer.Sound('Sound/crunch.wav')


	@property
	def direction(self):
		return self._direction
	
	@direction.setter
	def direction(self, val: Vector2):
		self.last_direction = self._direction
		self._direction = val

	def draw_snake(self):
		self.update_head_graphics()
		self.update_tail_graphics()

		for index,block in enumerate(self.body):
			x_pos = int(block.x * cell_size)
			y_pos = int(block.y * cell_size)
			block_rect = pygame.Rect(x_pos,y_pos,cell_size,cell_size)

			if index == 0:
				screen.blit(self.head,block_rect)
			elif index == len(self.body) - 1:
				screen.blit(self.tail,block_rect)
			else:
				previous_block = self.body[index + 1] - block
				next_block = self.body[index - 1] - block
				if previous_block.x == next_block.x:
					screen.blit(self.body_vertical,block_rect)
				elif previous_block.y == next_block.y:
					screen.blit(self.body_horizontal,block_rect)
				else:
					if previous_block.x == -1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == -1:
						screen.blit(self.body_tl,block_rect)
					elif previous_block.x == -1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == -1:
						screen.blit(self.body_bl,block_rect)
					elif previous_block.x == 1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == 1:
						screen.blit(self.body_tr,block_rect)
					elif previous_block.x == 1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == 1:
						screen.blit(self.body_br,block_rect)

	def update_head_graphics(self):
		head_relation = self.body[1] - self.body[0]
		if head_relation == Vector2(1,0): self.head = self.head_left
		elif head_relation == Vector2(-1,0): self.head = self.head_right
		elif head_relation == Vector2(0,1): self.head = self.head_up
		elif head_relation == Vector2(0,-1): self.head = self.head_down

	def update_tail_graphics(self):
		tail_relation = self.body[-2] - self.body[-1]
		if tail_relation == Vector2(1,0): self.tail = self.tail_left
		elif tail_relation == Vector2(-1,0): self.tail = self.tail_right
		elif tail_relation == Vector2(0,1): self.tail = self.tail_up
		elif tail_relation == Vector2(0,-1): self.tail = self.tail_down

	def move_snake(self):
		if self.new_block == True:
			body_copy = self.body[:]
			body_copy.insert(0,body_copy[0] + self.direction)
			self.body = body_copy[:]
			self.new_block = False
		else:
			body_copy = self.body[:-1]
			body_copy.insert(0,body_copy[0] + self.direction)
			self.body = body_copy[:]

	def add_block(self):
		self.new_block = True

	def play_crunch_sound(self):
		pass
		# self.crunch_sound.play()

	def reset(self):
		self.body = [Vector2(5,5),Vector2(4,5),Vector2(3,5)]
		self.direction = Vector2(1, 0) # RIGHT


class FRUIT:
	def __init__(self):
		self.randomize()

	def draw_fruit(self):
		fruit_rect = pygame.Rect(int(self.pos.x * cell_size),int(self.pos.y * cell_size),cell_size,cell_size)
		screen.blit(apple,fruit_rect)
		#pygame.draw.rect(screen,(126,166,114),fruit_rect)

	def randomize(self):
		self.x = random.randint(0,cell_number - 1)
		self.y = random.randint(0,cell_number - 1)
		self.pos = Vector2(self.x,self.y)

class MAIN:
	def __init__(self):
		self.snake = SNAKE()
		self.fruit = FRUIT()
		self.game_data: gameserializer.Game = []
		self.scores: list[int] = []

		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		print(f"Using device: {self.device}")

		self.early_model: torch.nn.Module = torch.load("model_early.pt", weights_only=False).to(self.device)
		
		if USE_MULTIPLE_MODELS:
			self.mid_model: torch.nn.Module = torch.load("model_mid.pt", weights_only=False).to(self.device)
			self.late_model: torch.nn.Module = torch.load("model_late.pt", weights_only=False).to(self.device)

		# self.loss_fn = torch.nn.MSELoss()
		# self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

		self.last_preds: torch.Tensor

	def update(self):
		self.snake.move_snake()
		self.check_collision()

		self.game_data.append(
			gameserializer.serialize_frame(
				food_pos=(self.fruit.x, self.fruit.y),
				snake_pos=[(int(body_piece.x), int(body_piece.y)) for body_piece in self.snake.body],
				board_size=(cell_number, cell_number),
				direction=self.snake.direction
			)
		)

		self.propagate_fail()

	def draw_elements(self):
		self.draw_grass()
		self.fruit.draw_fruit()
		self.snake.draw_snake()
		self.draw_score()

	def check_collision(self):
		if self.fruit.pos == self.snake.body[0]:
			self.fruit.randomize()
			self.snake.add_block()
			self.snake.play_crunch_sound()

		for block in self.snake.body[1:]:
			if block == self.fruit.pos:
				self.fruit.randomize()
	
	def has_failed(self):
		if not 0 <= self.snake.body[0].x < cell_number or not 0 <= self.snake.body[0].y < cell_number: return True

		for block in self.snake.body[1:]:
			if block == self.snake.body[0]: return True

	def propagate_fail(self):
		if self.has_failed():
			self.game_over()
			return
		
	def game_over(self):
		# train model

		# TRAIN_ENABLED = True

		# if TRAIN_ENABLED:
		# 	# move snake back to before death

		# 	DIRECTION_BEFORE_LOSING = self.snake.last_direction

		# 	self.snake.direction = -self.snake.direction
		# 	self.snake.move_snake()

		# 	self.snake.direction = DIRECTION_BEFORE_LOSING

		# 	# check squares around snake to find one that would not have resulted in death; choose those directions as the "correct" moves

		# 	correct_direction = [0, 0, 0, 0]

		# 	if self.propagate_bot_decision(0): # UP
		# 		self.snake.move_snake()

		# 		if not self.has_failed(): correct_direction[0] = 1

		# 		self.snake.direction = -self.snake.direction 
		# 		self.snake.move_snake()

		# 		self.snake.direction = DIRECTION_BEFORE_LOSING

		# 	if self.propagate_bot_decision(1): # RIGHT
		# 		self.snake.move_snake()

		# 		if not self.has_failed(): correct_direction[1] = 1

		# 		self.snake.direction = -self.snake.direction 
		# 		self.snake.move_snake()

		# 		self.snake.direction = DIRECTION_BEFORE_LOSING

		# 	if self.propagate_bot_decision(2): # DOWN
		# 		self.snake.move_snake()

		# 		if not self.has_failed(): correct_direction[2] = 1

		# 		self.snake.direction = -self.snake.direction 
		# 		self.snake.move_snake()

		# 		self.snake.direction = DIRECTION_BEFORE_LOSING

		# 	if self.propagate_bot_decision(3): # LEFT
		# 		self.snake.move_snake()

		# 		if not self.has_failed(): correct_direction[3] = 1

		# 		self.snake.direction = -self.snake.direction 
		# 		self.snake.move_snake()

		# 		self.snake.direction = DIRECTION_BEFORE_LOSING

		# 	if any(correct_direction): # only train if there is a non-losing direction
		# 		actual = torch.tensor(correct_direction, dtype=torch.float32).to(self.device)

		# 		loss = self.loss_fn(self.last_preds, actual)
		# 		loss.backward()
		# 		self.optimizer.step()

		if self.game_data and COLLECT_REFEED_TRAINING_DATA: gameserializer.serialize_game(self.game_data, 'refeed-games.dill')


		self.scores.append(game_proccessing_utils.get_score(self.game_data))

		self.snake.reset()
		self.game_data = []

		if len(self.scores) % 100 == 0:
			n = len(self.scores)
			average_score = sum(self.scores)/n
			highscore = max(self.scores)

			n_highscores = self.scores.count(highscore)

			print(f"Average Score over {n} games: {average_score}, Highscore: {highscore} (hit {n_highscores} times)")

	def draw_grass(self):
		grass_color = (167,209,61)
		for row in range(cell_number):
			if row % 2 == 0: 
				for col in range(cell_number):
					if col % 2 == 0:
						grass_rect = pygame.Rect(col * cell_size,row * cell_size,cell_size,cell_size)
						pygame.draw.rect(screen,grass_color,grass_rect)
			else:
				for col in range(cell_number):
					if col % 2 != 0:
						grass_rect = pygame.Rect(col * cell_size,row * cell_size,cell_size,cell_size)
						pygame.draw.rect(screen,grass_color,grass_rect)			

	def draw_score(self):
		score_text = str(len(self.snake.body) - 3)
		score_surface = game_font.render(score_text,True,(56,74,12))
		score_x = int(cell_size * cell_number - 60)
		score_y = int(cell_size * cell_number - 40)
		score_rect = score_surface.get_rect(center = (score_x,score_y))
		apple_rect = apple.get_rect(midright = (score_rect.left,score_rect.centery))
		bg_rect = pygame.Rect(apple_rect.left,apple_rect.top,apple_rect.width + score_rect.width + 6,apple_rect.height)

		pygame.draw.rect(screen,(167,209,61),bg_rect)
		screen.blit(score_surface,score_rect)
		screen.blit(apple,apple_rect)
		pygame.draw.rect(screen,(56,74,12),bg_rect,2)

	def bot_decision(self) -> None:
		flattened_board = game_proccessing_utils.flatten_board(
			gameserializer.serialize_frame(
					food_pos=(main_game.fruit.x, main_game.fruit.y),
					snake_pos=[(int(body_piece.x), int(body_piece.y)) for body_piece in main_game.snake.body],
					board_size=(cell_number, cell_number),
					direction=Vector2(0, 0)	
				).board
		)

		head_board, body_board, fruit_board = game_proccessing_utils.create_separate_boards(flattened_board)

		flat_board = torch.tensor(
			head_board+body_board+fruit_board,
			dtype=torch.float32
		).to(self.device)

		# self.model.train()
		# self.optimizer.zero_grad()

		body_length = len(self.snake.body)-1

		if USE_MULTIPLE_MODELS:
			if body_length <= EARLY_LENGTH_THRESH:
				model = self.early_model
				pygame.display.set_caption(f"using model: early")
			elif body_length <= MID_LENGTH_THRESH:
				model = self.mid_model
				pygame.display.set_caption(f"using model: mid")
			else:
				model = self.late_model
				pygame.display.set_caption(f"using model: late")
		else:
			model = self.early_model

		prediction: torch.Tensor = model(flat_board)

		self.last_preds = prediction

		vals, idxs = torch.topk(prediction, k=2)

		vals, idxs = (
			[val.item() for val in vals],
			[idx.item() for idx in idxs]
		)

		try:
			first_choice = random.choices(idxs, weights=vals, k=1)[0]

			self.propagate_bot_decision(first_choice) # if this fails, then just keep going in the same direction
		except ValueError: # model was indecisive; it outputted [0, 0, 0, 0]
			pass
	
	def propagate_bot_decision(self, choice: int) -> bool:
		if choice == 0: # UP
			if self.snake.direction.y != 1:
				self.snake.direction = Vector2(0, -1)
				return True
		if choice == 1: # RIGHT
			if self.snake.direction.x != -1:
				self.snake.direction = Vector2(1, 0)
				return True
		if choice == 2: # DOWN
			if self.snake.direction.y != -1:
				self.snake.direction = Vector2(0, 1)
				return True
		if choice == 3: # LEFT
			if self.snake.direction.x != 1:
				self.snake.direction = Vector2(-1, 0)
				return True

		return False

pygame.mixer.pre_init(44100,-16,2,512)
pygame.init()
cell_size = 40
cell_number = 10
screen = pygame.display.set_mode((cell_number * cell_size,cell_number * cell_size))
apple = pygame.image.load('Graphics/apple.png').convert_alpha()
game_font = pygame.font.Font('Font/PoetsenOne-Regular.ttf', 25)

SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, 1)

main_game = MAIN()

propagated_key_events: list[pygame.event.Event] = []
seen_key_event = False

STATS_MODE_ONLY = True

if STATS_MODE_ONLY: # removes graphics bottlenecks and simulates games internally (although the classes currently still hold internal Surface objects)
	print("Stats mode only enabled")
	pygame.quit()
	
	while True:
		main_game.bot_decision()
		main_game.update()

# otherwise, the visualizer here
while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()
		if event.type == SCREEN_UPDATE:
			main_game.bot_decision()
			main_game.update()
	
	screen.fill((175,215,70))
	main_game.draw_elements()
	pygame.display.update()