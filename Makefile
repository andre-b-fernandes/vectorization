autovectorized: simple.cpp
	g++ -O3 -mavx simple.cpp -o autovectorized

vectorized: vectorized.cpp
	g++ -mavx vectorized.cpp -o vectorized

simple: simple.cpp
	g++ simple.cpp -o simple

clean:
	rm vectorized autovectorized simple