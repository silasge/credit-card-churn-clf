ifeq ($(OS),Windows_NT)
	RM = del /Q
	FixPathIfWin = $(subst /,\,$1)
else
	RM = rm -f
	FixPathIfWin = $1
endif

RAW_PATH = ./data/raw
SRC_PATH = ./credit_card_churn_clf

clean_altair:
	$(RM) $(call FixPathIfWin, ./notebooks/*.json)
