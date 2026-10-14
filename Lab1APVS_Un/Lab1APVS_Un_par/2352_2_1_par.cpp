#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <vector>
#include <mpi.h>
using namespace std;

int main(int argc, char** argv) {
	// иницаиализация MPI
	MPI_Init(&argc, &argv); //передача аргументов командной строки в MPI

	int rank, size; // номер процесса и количество процессов
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // получение номера процесса в коммуникаторе MPI_COMM_WORLD
	MPI_Comm_size(MPI_COMM_WORLD, &size); // получение количества процессов в коммуникаторе MPI_COMM_WORLD

	int M = (argc >= 2) ? atoi(argv[1]) : 5000; //передача через mpiexec -n 4 ./program 10000 10, где 5000 - размер матрицы, 5 - количество запусков
	int runs = (argc >= 3) ? atoi(argv[2]) : 5;

	int base = M / size; // количество строк на процесс
	int remains = M % size;//количество оставшихся строк, которые нужно распределить между процессами, если M не делится на size нацело

	vector<int> rowEach(size); // количество строк на каждый процесс и выделение памяти для хранения этих значений
	vector<int> rowOffset(size);// смещение для каждого процесса, чтобы знать, с какой строки начинать обработку
	int offset = 0; // смещение для текущего процесса
	for (int i = 0; i < size; ++i) { 
		rowEach[i] = base + (i < remains ? 1 : 0); // распределение оставшихся строк между первыми remains процессами
		rowOffset[i] = offset; // сохранение смещения для текущего процесса
		offset += rowEach[i]; // обновление смещения для следующего процесса
	}

	vector<int> sendEach(size); // выделение памяти для хранения количества элементов, которые нужно отправить каждому процессу
	vector<int> sendOffset(size); // выделение памяти для хранения смещения для каждого процесса
	for (int i = 0; i < size; ++i) {
		sendEach[i] = rowEach[i] * M; // количество элементов, которые нужно отправить каждому процессу (количество строк * количество столбцов)
		sendOffset[i] = rowOffset[i] * M; // количество элементов, которые нужно пропустить перед отправкой данных каждому процессу (смещение * количество столбцов)
	}

	int localRows = rowEach[rank]; // количество строк, которые будет обрабатывать текущий процесс
	int localN = localRows * M; // количество элементов, которые будет обрабатывать текущий процесс (количество строк * количество столбцов)

	// выделение памяти для локальной матрицы A_local, вектора x и локального вектора y_local
	vector<double> A_local(localN);
	vector<double> x(M);
	vector<double> y_local(localRows);

	vector<double> A_full;
	vector<double> y_full;
	// выделение памяти для полной матрицы A_full и полного вектора y_full только на главный нулевой процесс
	if (rank == 0) {
		A_full.resize((size_t)M * M);
		y_full.resize(M);
	}
	// инициализация полной матрицы A_full и полного вектора x только на главном нулевом процессе
	if (rank == 0) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < M; j++) {
				A_full[(size_t)i * M + j] = i + j + 1.0;
			}
			x[i] = i + 1.0;
		}
	}
	// широковещательная рассылка вектора x всем процессам, чтобы каждый процесс имел доступ к полному вектору x
	MPI_Bcast(x.data(), M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// разбиение матрицы на части и распределение этих частей между процессами с помощью функции MPI_Scatterv
	MPI_Scatterv(rank == 0 ? A_full.data() : nullptr, sendEach.data(), sendOffset.data(), MPI_DOUBLE,
				 A_local.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double best = 1e18; // обозначение для хранения лучшего времени выполнения, инициализированное очень большим числом 1*10^18
	for (int r = 0; r < runs; r++) {
		MPI_Barrier(MPI_COMM_WORLD); // синхронизация всех процессов перед началом измерения времени
		double t0 = MPI_Wtime(); // Начало измерения времени выполнения

		for (int i = 0; i < localRows; ++i) {
			double sum = 0.0;
			for (int j = 0; j < M; ++j) {
				sum += A_local[(size_t)i * M + j] * x[j]; // вычисление скалярного произведения строки матрицы A_local и вектора x
			}
			y_local[i] = sum;
		}

		MPI_Barrier(MPI_COMM_WORLD); // синхронизация всех процессов после завершения вычислений
		double t1 = MPI_Wtime(); // Конец измерения времени выполнения
		if (t1 - t0 < best) best = t1 - t0; // обновление лучшего времени выполнения, если текущее время меньше предыдущего лучшего времени
	}

	MPI_Gatherv(y_local.data(), localRows, MPI_DOUBLE,
		rank == 0 ? y_full.data() : nullptr, rowEach.data(), rowOffset.data(), MPI_DOUBLE,
		0, MPI_COMM_WORLD); // сбор локальных результатов y_local в полный вектор y_full на главном процессе

	if (rank == 0) {
		cout << "MPI matvec, M = " << M << ", size = " << size << ", runs = " << runs << "\n\n";
		cout << fixed << setprecision(7);
		cout << "Best time: " << best << " s\n\n";

		cout << setprecision(2);
		cout << "y[0] = " << y_full[0] << "\n";
		cout << "y[M/2] = " << y_full[M / 2] << "\n";
		cout << "y[M-1] = " << y_full[M - 1] << "\n";
	}

	MPI_Finalize();
	return 0;
}
