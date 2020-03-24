ARGS=preprocess.py

clean:
	rm \#*
	rm *\#
	rm *~

run:
	python $(ARGS)
