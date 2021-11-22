// normal_distribution
#include <iostream>
#include <string>
#include <random>
#include <pthread.h>
#include <omp.h>

using namespace std;

struct thread_data
{
	double * values;
	int start_index;
	int end_index;
	float result;
	int key;
};

void * calc_sum (void* args)
{

	thread_data * arguments = ((thread_data*)args);	
	//cout<<"thread with "<< arguments->start_index << "started"<<endl;
	double result = 0;
	for (int i = arguments->start_index ; i < arguments->end_index;i++)
	{
		for (int k = 0; k < 150000; k++)
		result += 1;
		//cout<<arguments->key<<endl;
	}
	//arguments->result = result;
	//cout<<result<<endl;
}

int main()
{
	//cout<<"all data  is being created"<<endl;

default_random_engine generator(time(0));
normal_distribution<double> distribution(0,1);
int num_threads = 50;
//cout<<"please enter number of threads"<<endl;
//cin>>num_threads;

//cout<<"all data  is being created"<<endl;

int all_data_size = 400;
//cout<<"please enter the data size"<<endl;

pthread_t threads[num_threads];
thread_data data_for_threads[num_threads];
int slice_size = all_data_size / num_threads;
double data[all_data_size];
//cout<<"all data  is being created"<<endl;
for(int i = 0; i<all_data_size;i++)
{
	data[i] = 0.00001;
}

for (int t = 0; t<num_threads;t++)
{
	//cout<<"data for thread is being created"<<endl;
	data_for_threads[t].start_index = t * slice_size;
	data_for_threads[t].end_index = (t+1) * slice_size;
	data_for_threads[t].values = data;
	data_for_threads[t].key = t;

}
//const clock_t begin_time = clock();
double runTime = -omp_get_wtime();


for(int t = 0 ; t<num_threads;t++)
{	
	//cout<<"thread is being created"<<endl;
    pthread_create(&(threads[t]), NULL, calc_sum, (void*)(&(data_for_threads[t])));
}


//cout<<"joining threads"<<endl;

for(int t = 0; t<num_threads;t++)
{
    pthread_join(threads[t], NULL);
}
//cout<<"threads joined";

//cout<<"elapsed time"<<endl;
runTime += omp_get_wtime();
cout<<runTime<<endl;
float result = 0;
for (int t = 0; t<num_threads;t++)
{
	result+= data_for_threads[t].result;
}



//cout<<endl<<result;
return(0);
}
