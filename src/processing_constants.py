MIN_SCORE_THRESH = 8 # minimum score threshold to include a game for processing

USE_MULTIPLE_MODELS = False # whether or not to use multiple models for different game stages

# parameters for refeed training (feeding the model's own gameplay back into training data)
USE_REFEED_TRAINING = False
COLLECT_REFEED_TRAINING_DATA = False

# thresholds for splitting gameplay into early, mid, and late stages
EARLY_LENGTH_THRESH = 10
MID_LENGTH_THRESH = 25