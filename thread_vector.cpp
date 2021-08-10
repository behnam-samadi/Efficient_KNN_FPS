#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <vector>

using namespace std;

void *print_message_function(void *a);

struct arg_to_thread{
     int size;
     int start_value;
     char message;

};

int main()
{
     vector<pthread_t> threads;
     pthread_t thread;
     threads.push_back(thread);
     threads.push_back(thread);
     arg_to_thread arg1;
     arg_to_thread arg2;
     arg1.size = 100000;
     arg2.size = 100000;
     arg1.start_value = -100000;
     arg2.start_value = 50000;
     arg1.message = 'F';
     arg2.message = 'S';


     int message1[10000];
     int message2[10000];
     int  iret1, iret2;

    /* Create independent threads each of which will execute function */

     iret1 = pthread_create( &(threads[0]), NULL, print_message_function, (void*)&arg1);
     cout<<"p";
     iret2 = pthread_create( &(threads[1]), NULL, print_message_function, (void*)&arg2);

     /* Wait till threads are complete before main continues. Unless we  */
     /* wait we run the risk of executing an exit which will terminate   */
     /* the process and all threads before the threads have completed.   */

     pthread_join( threads[0], NULL);
     pthread_join( threads[1], NULL); 

     printf("Thread 1 returns: %d\n",iret1);
     printf("Thread 2 returns: %d\n",iret2);
     exit(0);
}

void* print_message_function(void* a )
{
     arg_to_thread args = *((arg_to_thread*)a);
     for (int i= args.start_value;i<args.start_value + args.size;i++){
          cout<<args.message<<" ";

     }


}
