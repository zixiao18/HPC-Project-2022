all:
	g++ -Wall -std=c++11 main.cpp utility.cpp lr.cpp -o main 

clean:
	rm main

test:
	./main -t wine_train.csv -p wine_test.csv -nr 3526 -nc 11 -nl 7 -tr 882 -tl 11 -i 100 -a 0.001 -l 0.005
