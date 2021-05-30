#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include "numeric"
using namespace std;


void print_vector (vector<int> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<v[i]<<" ";
    }
    cout<<endl;
}

int main()
{
	//Assume A is a given vector with N elements
	int N = 5;
	vector<int> A {3,5,4,1,2};
	vector<int> V(N);
	iota(V.begin(),V.end(),0); //Initializing
	sort( V.begin(),V.end(), [&](int i,int j){return A[i]<A[j];} );
	sort( A.begin(),A.end());
	print_vector(V);
	cout<<endl;
	print_vector(A);

}