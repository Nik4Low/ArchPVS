#include <iostream>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <vector> // Подключаем заголовочный файл для использования std::vector
using namespace std;

//Функция генерации матрицы A и вектора x
void generate(double* A, double* x, int M) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			A[(size_t)i * M + j] = i + j + 1.0;
		}
		x[i] = i + 1.0;
	}
}

//Функция умножения матрицы A на вектор x с результатом в векторе y
void matvec( double* A, double* x, double* y, int M) {
	for (int i = 0; i < M; i++) {
		double sum = 0.0;
		for (int j = 0; j < M; j++) {
			sum += A[(size_t)i * M + j] * x[j];
		}
		y[i] = sum;
	}
}

int main(int argc, char* argv[]) {
	int M = 1000; // Размерность матрицы и вектора
	int runs = 5; // Число итераций для измерения времени выполнения

	if (argc >= 2) {
		M = std::atoi(argv[1]);
	}
	if (argc >= 3) {
		runs = std::atoi(argv[2]);
	}
	cout << "Sequential matvec\n";
	cout << "Matrix size: " << M << "x" << M << ", runs: " << runs << "\n\n";

	// Выделение памяти для матрицы A, вектора x и вектора y
	vector<double> A((size_t)M * M);
	vector<double> x(M);
	vector<double> y(M);
	
	// Генерация матрицы A и вектора x
	generate(A.data(), x.data(), M);

	// Измерение времени для умножения матрицы на вектор
	double best = 1e18;
	for (int r = 0; r < runs; r++) {
		auto t0 = chrono::high_resolution_clock::now();
		matvec(A.data(), x.data(), y.data(), M);
		auto t1 = chrono::high_resolution_clock::now();
		double t = chrono::duration<double>(t1 - t0).count();
		if (t < best) best = t;

		cout << "Run " << r + 1 << ": " << fixed << setprecision(8) << t << " seconds\n";

	}

	cout << "Best time: " << fixed << setprecision(8) << best << " seconds\n";

	// Вывод результатов для проверки корректности
	cout << "Result y[0]: " << setprecision(2) << y[0] << "\n";
	cout << "Result y[M/2]: " << y[M / 2] << "\n";
	cout << "Result y[M-1]: " << y[M - 1] << "\n";



	return 0;
}


