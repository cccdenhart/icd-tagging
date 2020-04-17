SUB=""
LIMIT=""
MODEL="baseline"
EMBEDDINGS=""

clean:
	rm -rf slurm*
	rm -rf \#*
	rm -rf *\#
	rm -rf ~*

preprocess:
	python -m build --roots --notes --prep $(SUB)

split:
	python -m build --split $(LIMIT)

train:
	python -m build --$(MODEL) --$(EMBEDDINGS)
