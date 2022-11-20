#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

using namespace std;
using namespace std::chrono;


const int array_size = 8*200;

void showMatrix(float* result){
    cout << endl << "Matrix: " << endl;
    for(int i = 0; i < array_size; ++i)
        for(int j = 0; j < array_size; ++j){
            cout << " " << result[i* array_size + j];
                if(j == array_size - 1)
                        cout << endl;
        }
}


void vectorizedMatrixMultiplication4(float* first, float* second, float* result){
    for(int i = 0; i < array_size; i++){
        for(int j = 0; j < array_size; j+=4){
            __m128 row = _mm_setzero_ps();
            for(int k = 0; k < array_size; k++ ){
                // fetches the element referenced by the pointer and broadcasts it into a vectorized scaled single precision.
                // Since its row X column, the each row element will have to be multiplied by each member of the column.
                __m128 first_packed = _mm_broadcast_ss(first + i * array_size + k);
                // row of the second matrix.
                __m128 second_packed = _mm_load_ps(second + k*array_size + j);
                __m128 multiplied_packed = _mm_mul_ps(first_packed, second_packed);
                row = _mm_add_ps(row, multiplied_packed);
            } 
            _mm_store_ps(result + i*array_size + j, row);
        }
    }
}

void vectorizedMatrixMultiplication8(float* first, float* second, float* result){
    for(int i = 0; i < array_size; i++){
        for(int j = 0; j < array_size; j+=8){
            __m256 row = _mm256_setzero_ps();
            for(int k = 0; k < array_size; k++ ){
                // fetches the element referenced by the pointer and broadcasts it into a vectorized scaled single precision.
                // Since its row X column, the each row element will have to be multiplied by each member of the column.
                __m256 first_packed = _mm256_broadcast_ss(first + i * array_size + k);
                // row of the second matrix.
                __m256 second_packed = _mm256_load_ps(second + k*array_size + j);
                __m256 multiplied_packed = _mm256_mul_ps(first_packed, second_packed);
                row = _mm256_add_ps(row, multiplied_packed);
            } 
            _mm256_store_ps(result + i*array_size + j, row);
        }
    }
}


void vectorized4(float* randArray, float* randArray2, float* result){
    auto start = high_resolution_clock::now();
    vectorizedMatrixMultiplication4(randArray, randArray2, result);
    auto finish = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(finish - start);
    cout << "Vectorized Matrix Multiplication(4) took: " << duration.count() << " miliseconds to execute." << endl;
}

void vectorized8(float* randArray, float* randArray2, float* result){
    auto start = high_resolution_clock::now();
    vectorizedMatrixMultiplication8(randArray, randArray2, result);
    auto finish = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(finish - start);
    cout << "Vectorized Matrix Multiplication(8) took: " << duration.count() << " miliseconds to execute." << endl;
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

    //showMatrix(randArray);
    //showMatrix(randArray2);

    vectorized4(randArray, randArray2, result);
    vectorized8(randArray, randArray2, result);
    //showMatrix(result);

    free(randArray);
    free(randArray2);
    free(result);

    return 0;
}
