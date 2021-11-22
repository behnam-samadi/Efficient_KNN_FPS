#include <iostream>
#include<fstream>
#include<vector>

using namespace std;

int main()
{
	string adress;
	cout<<"enter the file adress ";
	cin>>adress;
	ofstream fout(adress, std::ios::app);
	fout<<",e";
	fout.close();



}