.PHONY: setup lesson1 lesson2 lesson3 lesson4 lesson5 lesson6 test clean

setup:
	uv sync

lesson1:
	uv run python lessons/code/01_tokenization_and_data.py

lesson2:
	uv run python lessons/code/02_bigram_baseline.py

lesson3:
	uv run python lessons/code/03_self_attention.py

lesson4:
	uv run python lessons/code/04_multi_head_attention.py

lesson5:
	uv run python lessons/code/05_transformer_block.py

lesson6:
	uv run python lessons/code/06_full_gpt_and_training.py

test:
	uv run python -m pytest tests/ -v

clean:
	rm -rf __pycache__ lessons/code/__pycache__ .pytest_cache
