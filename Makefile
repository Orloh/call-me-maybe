.PHONY install run debug clean lint lint-strict

NAME = src

RUN = uv run

all: run

install:
	uv sync

run: clear && $(RUN) -m $(NAME)

debug:
	$(RUN) -m pdb -m $(NAME)

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .mypy_cache dist
	rm -rf data/output/*

lint:
	$(RUN) flake8 $(NAME)
	$(RUN) mypy $(NAME) . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
