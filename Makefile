CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
PYTHON   ?= python3

.PHONY: build test experiments clean

build: libengine.so

libengine.so: engine.cpp
	$(CXX) $(CXXFLAGS) -pthread -shared -fPIC engine.cpp -o libengine.so

test: build
	$(PYTHON) tests/test_pricer.py

experiments: build
	$(PYTHON) run_experiments.py

clean:
	rm -f libengine.so
