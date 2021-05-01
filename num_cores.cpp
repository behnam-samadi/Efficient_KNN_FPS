#include <unistd.h>
#include <iostream>
using namespace std;

int main(){
int number_of_cores = sysconf(_SC_NPROCESSORS_ONLN);
cout<<number_of_cores;
return (0);
}
