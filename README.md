run main.py out of the box, after installing the requirements:

1) make sure you're using python 3.7. Tensorflow 1.15 is needed for me to stop bugs on the M1 chip
and its only compatible with 3.7 at the latest.

main.py runs the main training code, and it should use environment-DepotEarlyTerm by default - this is the environment
that the Depp Q Learning uses.