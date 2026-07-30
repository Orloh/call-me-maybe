# VARIABLES

SRC = src

RUN = uv run python

MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

CHECK_UV = command -v uv

INSTALL_UV = curl -LsSF https://astral.sh/uv/install.sh | sh

CPU ?= 0
ifeq ($(CPU), 1)
    CUDA_PREFIX = CUDA_VISIBLE_DEVICES=""
else
    CUDA_PREFIX =
endif


# RULES
.PHONY: all install run debug trace test test-all clean lint lint-strict

all: install run

install:
	@if !$(CHECK_UV) > /dev/null 2>&1; then \
		echo "$(BRED)UV not installed. Installing...$(RESET)"; \
		$(INSTALL_UV); \
	fi
	@echo "$(BGREEN)Installing project dependencies using uv...$(RESET)"; \
	
	uv sync --link-mode copy

run: install
	clear && $(CUDA_PREFIX) $(RUN) -m $(SRC)

debug: install
	clear
	@echo "$(BGREEN)Running the main script in debug mode (pdb)...$(RESET)"
	$(CUDA_PREFIX) $(RUN) -m pdb -m $(SRC) --trace

trace:
	clear
	@echo "$(BGREEN)Running the main script with per-token PDA/FSM tracing...$(RESET)"
	$(CUDA_PREFIX) $(RUN) -m $(SRC) --trace

test:
	$(RUN) -m pytest

test-all:
	$(RUN) -m pytest -m benchmark -s

clean:
	@echo "$(YELLOW)Cleaning temporary files and caches...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .mypy_cache
	rm -rf .pytest_cache

lint:
	@clear
	@echo "$(BMAGENTA) Running standard linting...$(RESET)"
	$(RUN) -m flake8 $(SRC)
	$(RUN) -m mypy $(SRC) $(MYPY_FLAGS)


# COLORS
RESET		=	\033[0m
BGREEN		=	\033[92m
BMAGENTA	=	\033[95m
YELLOW		=	\033[93m
BRED		=	\033[91m
