#include <iostream>
#include <pthread.h>
using namespace std;


struct thread_data
{
	int * data;
	int start_index;
	int end_index;
};


void * thread_func (void * args_void)
{
	thread_data * args = (thread_data*)(args_void);
	for (int i = args->start_index ; i <= args->end_index; i++)
	{
		args->data[i] = 2 * args->data[i];
	}
}


int main()
{
	int test_size = 1024000;
	int * data = new int[test_size];
	for (int i = 0 ; i < test_size ;i++) data[i]  = 1;
	int num_threads = 1;
	int slice_size = test_size / num_threads;
	thread_data * data_for_threads = new thread_data [num_threads];
	for (int i = 0 ; i < num_threads ; i++)
	{
		data_for_threads[i].data = data;
		data_for_threads[i].start_index = i*slice_size;
		data_for_threads[i].end_index = (i+1)*slice_size - 1;
	}
	pthread_t threads[num_threads];
	for(int t = 0 ; t<num_threads;t++)
	{	
    	pthread_create(&(threads[t]), NULL, thread_func, (void*)(&(data_for_threads[t])));
	}
	for(int t = 0; t<num_threads;t++)
	{
    pthread_join(threads[t], NULL);
	}
	cout<<endl<<"parlle computations is done"<<endl;
	for (int i = 0 ; i<test_size;i++)
	{
		if (data[i] != 2)
		{
			cout<<endl<<"error!"<<endl;
			break;
		}
	}
	cout<<"ok!";
}