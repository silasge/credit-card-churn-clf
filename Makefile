all: poetry_install_env predict

.PHONY: poetry_install_env

# Diretórios
RAW_DATA = ./data/raw/credit_card_churn.csv
SPLITS_PATH = ./data/splits
PREDICTIONS_DATA = ./data/predictions/predictions.pkl
MODELS_PATH = ./models
SRC_PATH = ./credit_card_churn_clf
SRC_SCRIPTS_PATH = $(SRC_PATH)/scripts
SRC_DATA_PATH = $(SRC_PATH)/data
SRC_MODELS_PATH = $(SRC_PATH)/models

# Poetry run
POETRY_RUN = poetry run python

# Parâmetros
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV = 5
N_ITER = 50
SCORING = roc_auc
THRESHOLD = 0.22

# Modelos
MODELS = $(MODELS_PATH)/LogisticRegression.pkl \
         $(MODELS_PATH)/DecisionTreeClassifier.pkl \
		 $(MODELS_PATH)/RandomForestClassifier.pkl \
		 $(MODELS_PATH)/GradientBoostingClassifier.pkl

# Comandos que dependem do OS
ifeq ($(OS),Windows_NT)
	RM = del /Q
	FixPathIfWin = $(subst /,\,$1)
else
	RM = rm -f
	FixPathIfWin = $1
endif

poetry_install_env:
	@poetry install
	
read_and_split: $(SPLITS_PATH)/credit_card_churn_train_test.pkl
train: $(MODELS)
predict: $(PREDICTIONS_DATA)

$(SPLITS_PATH)/credit_card_churn_train_test.pkl: $(SRC_SCRIPTS_PATH)/read_and_split.py $(RAW_DATA) $(SRC_DATA_PATH)/data_funcs.py
	@$(POETRY_RUN) $< --csv $(RAW_DATA) \
	                  --test_size $(TEST_SIZE) \
					  --random_state $(RANDOM_STATE) \
					  --export_pkl_to $@ \
					  --print_log

$(MODELS): $(SRC_SCRIPTS_PATH)/train_models.py \
           $(SPLITS_PATH)/credit_card_churn_train_test.pkl \
		   $(SRC_MODELS_PATH)/feature_engineering.py \
		   $(SRC_MODELS_PATH)/modeling_funcs.py
	@$(POETRY_RUN) $< --data $(word 2, $^) \
	                  --model $(notdir $(basename $@)) \
					  --cv $(CV) \
					  --scoring $(SCORING) \
					  --n_iter $(N_ITER) \
					  --random_state $(RANDOM_STATE) \
					  --export_pkl_to $@ \
					  --print_log

$(PREDICTIONS_DATA): $(SRC_SCRIPTS_PATH)/predict_test_set.py \
                     $(SPLITS_PATH)/credit_card_churn_train_test.pkl \
					 $(MODELS) \
                     $(SRC_MODELS_PATH)/feature_engineering.py \
		             $(SRC_MODELS_PATH)/modeling_funcs.py
	@$(POETRY_RUN) $< --data $(word 2, $^) \
	                  --threshold $(THRESHOLD) \
					  --models_path $(MODELS) \
					  --export_pkl_to $@

clean:
	@$(RM) $(call FixPathIfWin, $(SPLITS_PATH)/*.pkl)
	@$(RM) $(call FixPathIfWin, $(PREDICTIONS_DATA))

clean_altair:
	@$(RM) $(call FixPathIfWin, ./notebooks/*.json)

black:
	@poetry run black $(SRC_PATH)
