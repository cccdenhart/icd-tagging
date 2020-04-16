ARGS=preprocess.py

clean:
	rm -rf slurm*
	rm -rf \#*
	rm -rf *\#
	rm -rf ~*

run:
	python $(ARGS)
