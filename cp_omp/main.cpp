#include <iostream>
#include <omp.h>

using namespace std;

long num_steps = 100000;

double pi_sequential()
{
	double x, sum = 0.0;
	double step = 1.0 / (double)num_steps;

	for (long i = 1; i <= num_steps; i++) 
	{
		x = (i - 0.5) * step;
		sum += 4.0 / (1.0 + pow(x, 2.0));
	}

	//cout << "Sequential sum = " << sum << '\n';
	double pi = step * sum;
	return pi;
}

double pi_parallel(const unsigned int noOfThreads)
{
	const double step = 1.0 / (double)num_steps;

	double *buffer;
	buffer = (double*)malloc(noOfThreads * sizeof(double));
	for (int i = 0; i < noOfThreads; i++)
	{
		buffer[i] = 0.0;
	}

	#pragma omp parallel num_threads (noOfThreads)
	{
		double x;
		const int thread_num = omp_get_thread_num();
		double increment = 0.0;

		for (long i = thread_num + 1; i <= num_steps; i += noOfThreads)
		{
			x = (i - 0.5) * step;
			increment += 4.0 / (1.0 + pow(x, 2.0));
		}

		buffer[thread_num] = increment;
		//printf("Thread %d = %f\n", thread_num, increment);
	}

	double sum = 0.0;
	for (int i = 0; i < noOfThreads; i++)
		sum += buffer[i];

	//printf("Concurent sum = %f\n", sum);
	double pi = step * sum;

	free(buffer);
	return pi;
}

double pi_parallel_WSC(const unsigned int noOfThreads)
{
	double x, sum = 0.0;
	double step = 1.0 / (double)num_steps;

	#pragma omp parallel for num_threads(noOfThreads)
	for (long i = 1; i <= num_steps; i++)
	{
		x = (i - 0.5) * step;
		sum += 4.0 / (1.0 + pow(x, 2.0));
	}

	double pi = step * sum;
	return pi;
}

double pi_parallel_shared(const unsigned int noOfThreads)
{
	double x, sum = 0.0;
	double step = 1.0 / (double)num_steps;

	#pragma omp parallel for num_threads(noOfThreads) private(x) shared(sum)
	for (long i = 1; i <= num_steps; i++)
	{
		x = (i - 0.5) * step;
		#pragma omp critical
		sum += 4.0 / (1.0 + pow(x, 2.0));
	}

	double pi = step * sum;
	return pi;
}

int main() 
{
	cout << "Sequential = " << pi_sequential() << '\n';
	cout << "Parallel = " << pi_parallel(4) << '\n';
	cout << "Parallel WSC = " << pi_parallel(4) << '\n';
	cout << "Parallel Shared = " << pi_parallel_shared(4) << '\n';
	
	char x;
	cin >> x;
	return 0;
}