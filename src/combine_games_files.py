import os
import dill
import gameserializer

# combine all game data
games: gameserializer.Games = []

for file in os.listdir():
	if os.path.isfile(file) and file.startswith("games") and file.endswith(".dill"):
		games.extend(gameserializer.read_games(file))

dill.dump(games, open('games.dill', 'wb'))