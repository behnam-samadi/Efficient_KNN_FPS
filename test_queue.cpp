#include <iostream>
#include <queue>
#include <bits/stdc++.h>

using namespace std;

int main()
{
	priority_queue<int> q;
	priority_queue<pair<int, int>> pq;
	q.push(90);
	q.push(91);
	q.push(19);
	q.push(20);
	cout<<endl<<q.top();
	q.pop();
	cout<<endl<<q.top();
}