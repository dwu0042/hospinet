docs:
	pdoc --force --output-dir docs --html tempest

test:
	pytest -v 