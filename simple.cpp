#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace std::chrono;


const int array_size = 8*200;

void matrixMultiplication(float* first, float* second, float* result){
    for(int i = 0; i < array_size; ++i){
        for(int j = 0; j < array_size; ++j){
            result[i*array_size + j] = 0;
            for(int k = 0; k < array_size; ++k){
                result[i*array_size + j] += first[i*array_size + k] * second[k*array_size + j];
            } 
        }
    }
}

void simple(float* randArray, float* randArray2, float* result){
    auto start = high_resolution_clock::now();
    matrixMultiplication(randArray, randArray2, result);
    auto finish = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(finish - start);
    
    cout << "Simple Matrix Multiplication took: " << duration.count() << " miliseconds to execute." << endl;
}

int main(){

    float* randArray = (float*) aligned_alloc(32, array_size * array_size * sizeof(float));
    float* randArray2 = (float*) aligned_alloc(32, array_size * array_size * sizeof(float));
    float* result = (float*) aligned_alloc(32, array_size * array_size * sizeof(float));

    for(int i = 0; i < array_size; ++i){
        for(int j = 0; j < array_size; ++j){
            randArray[i*array_size + j] = static_cast <float>(rand() % 10);
            randArray2[i*array_size + j] = static_cast <float>(rand() % 10);
        }
    }

    simple(randArray, randArray2, result);

    free(randArray);
    free(randArray2);
    free(result);
    
}