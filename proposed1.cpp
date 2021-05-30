#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include<cmath>
#include <pthread.h>
#include "numeric"
using namespace std;

class Frame{
    public:
    int num_points;
    int points_dim;
    vector<vector<float>> data;
};

struct thread_data
{
    vector<int>* sorted_indices;
    vecotr<float>* sum_cordinates;
    int begin_index;
    int end_index;
    vector<vector<float>> points;
};



float calc_distance (vector<float> v1, vector<float> v2, string type)
{
    
    if (type == "Modified_Manhattan")
    {
    	float sum1 = 0;
    	float sum2 = 0;
    	for(int i = 0; i<v1.size();i++)
    		sum1+=v1[i];
    	for(int i = 0; i<v2.size();i++)
    		sum2+=v2[i];
    	return (abs(sum2 - sum1));
    }
    else
    {
    	float sum = 0;
    for(int i = 0; i<v1.size();i++)
    {
        if (type=="Euclidean")
        sum+= pow(abs(v1[i] - v2[i]), 2);
        if (type=="Manhattan")
        sum+= abs(v1[i] - v2[i]);
        //cout<<"sum become:"<<sum<<"\n";
    }
    //cout<<"\n"<<"sum:"<<sum;

    float result = sum;
    if (type == "Euclidean")
        result = sqrt(result);
    return(result);
    }
}


vector<int> topK(vector<float> input, int K){
    float inf = 0;
    for (int i = i ; i< input.size();i++)
    {
        if (input[i] > inf) inf = i;
    }
    inf = inf + 100;

    vector<int> result(K);
    for (int c = 0; c<K; c++){
        int min_arg = 0;
        for (int j = 0; j<input.size();j++)
        {
            if(input[j] < input[min_arg]){
                min_arg = j;
            }
        }
        //cout<<"maxarg: " <<min_arg<<"\n";
        result[c]  = min_arg;
        input[min_arg] = inf;

    }
return (result);
}

Frame read_data (string file_adress, int points_dim, int output_dims)
{ 
    ifstream fin(file_adress, ios::binary);
    fin.seekg(0, ios::end);
    const size_t num_elements = fin.tellg() / sizeof(float);
    fin.seekg(0, ios::beg);
    Frame frame;
    frame.points_dim = output_dims;
    frame.num_points = num_elements/points_dim;
    vector<float> data_temp(num_elements);
    vector<vector<float>> data  (num_elements/points_dim , vector<float> (output_dims, 0));
    fin.read(reinterpret_cast<char*>(&data_temp[0]), num_elements*sizeof(float));
    for (size_t i = 0; i < frame.num_points; i++){
    for(size_t j = 0; j < frame.points_dim; j++)
    {
        data[i][j] = data_temp[i*points_dim + j];
    }
    
}
	frame.data = data;
	return(frame);
}

vector<vector<int>> KNN (Frame reference, Frame query, int K,string metric,  int num_query=0){
    int num_ref_points = reference.data.size();
    int num_query_points = query.data.size();
    //imidiate
    //num_ref_points = 12;
    if (!(num_query == 0)) num_query_points = num_query;
    vector<vector<int>> result  (num_query_points , vector<int> (K, 0));
    vector<float>  distance (num_ref_points);

    for(int i = 0; i<num_query_points;i++){
        cout<<"KNN, Progress:" <<(float)i/num_query_points<<"\n";
        for (int j = 0; j<num_ref_points;j++)
        {
            distance[j] = calc_distance(query.data[i], reference.data[j], metric);
        }
        vector<int> topk = topK(distance, K);
        for(int c = 0; c<K;c++)
        {
            result[i][c] = topk[c];
        }
        
    }
return(result);
}

void print_vector (vector<int> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<v[i]<<" ";
    }
    cout<<endl;
}

void print_vector_2D (vector<vector<int>>input){
    for (int i = 0; i< input.size();i++)
    {
        for(int j = 0; j<input[0].size();j++)
        {
            cout<<input[i][j]<<" ";
        }
        cout<<endl;
    }

}

void * thread_function(void *)
{

}

struct query_point
{
    vector<float> point;
    
}

int main()
{
	int frame_channels = 3;
    Frame reference = read_data("0000000000.bin", 4, frame_channels);
    Frame query = read_data("0000000001.bin", 4, frame_channels);
    int num_ref_points = reference.data.size();
    int num_query_points = query.data.size();
    num_query_points = 512;
    int round_size = 64;
    cout<< num_ref_points<<" " << num_query_points<<endl;
    vector<float> sum_cordinates(num_ref_points);
    for (int i =0 ; i<num_ref_points;i++)
    {
        sum_cordinates[i] = 0;
        //cout<<endl<<endl;
        for (int j = 0; j<reference.data[0].size();j++)
        {
        sum_cordinates[i] += reference.data[i][j];
        //cout<<"sum_cordinates "<< sum_cordinates[i]<<" ,"<<"reference "<< reference.data[i][j]<<endl;
        }
    }

    vector<int> sorted_indices(num_ref_points);
    iota(sorted_indices.begin(),sorted_indices.end(),0); //Initializing
    sort( sorted_indices.begin(),sorted_indices.end(), [&](int i,int j){return sum_cordinates[i]<sum_cordinates[j];} );
    sort( sum_cordinates.begin(),sum_cordinates.end());


    //print_vector(sum_cordinates);
    print_vector(sorted_indices);
    int num_round = num_query_points / round_size;
    for (int round = 0 ; round < num_round;round++)
    {
        
    }

    exit(0);
    int k = 40;
    vector<vector<int>> knn_result = KNN(reference, query, k, "Modified_Manhattan",num_query_points);
    cout<<knn_result.size()<<endl;
    cout<<knn_result[0].size();
    print_vector_2D(knn_result);
    vector<vector<int>>  ground_truth = KNN(reference, query, k, "Euclidean",num_query_points);
    int mathces = 0;
    for (int i = 0; i<num_query_points;i++)
    {
    	for (int j = 0 ; j < k ; j++)
    	{
    		bool found = false;
    		for (int c= 0; c<k;c++)
    		{
    			if(knn_result[i][j]==ground_truth[i][c])
    			{
	    			found = true;
	    			break;
    			}
    		}
    		if (found == true) mathces++;
    	}
    }
    cout<<(float)mathces/((float)num_query_points*(float)k);

    //vector<float> test2{10, 20 , 30};
    //vector<float> test1{110, 120 , 130};
    //cout<<calc_distance(test1, test2, "Modified_Manhattan");
    //for (int i = 0 ; i<num_ref_points;i++)cout<<reference.data[i][0]<<","<<reference.data[i][1]<<","<<reference.data[i][2]<<endl;
}