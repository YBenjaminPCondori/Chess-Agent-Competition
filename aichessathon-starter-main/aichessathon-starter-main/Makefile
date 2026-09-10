SHELL := /bin/bash

.PHONY: setup play arena live zip gate

OPPONENT ?= baselines/greedy
GAMES ?= 16
BASE_MS ?= 10000
INCREMENT_MS ?= 100

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black $(OPPONENT) $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent $(OPPONENT) --games $(GAMES) --base-ms $(BASE_MS) --increment-ms $(INCREMENT_MS)

live: arena

zip:
	uv run python -m harness.package

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent $(OPPONENT) --games $(GAMES) --base-ms $(BASE_MS) --increment-ms $(INCREMENT_MS)
