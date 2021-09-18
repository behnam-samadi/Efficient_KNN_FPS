#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include<cmath>
#include <pthread.h>
#include <queue>
#include <limits>
#include "numeric"
#include <limits>
#include <omp.h>
#include <time.h>
#define Points_Dim 3
using namespace std;


class Frame{
    public:
    int num_points;
    int points_dim;
    vector<vector<float>> data;
};

struct node_boundries
{
    //vector<vector<bool>> is_set (3 , vector<bool> (2, 0));
    //vector<vector<float>> limits (3 , vector<float> (2, 0));
    vector<bool> is_set;
    vector<vector<float>> limits;
};

void print_vector (vector<bool> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<v[i]<<" ";
    }
    cout<<endl;
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

void print_vector_float (vector<float> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<v[i]<<" ";
    }
    cout<<endl;
}

struct node
{
	int dimension;
	float branchpoint;
	//float borders[Points_Dim][2];
    node_boundries boundries;
    vector<float> point;
};


void create_kd_tree_rec(node* tree, int index, int dimension, vector<vector<float>> *all_points, vector<int>sub_points_indices, node_boundries boundries)
{
    if (sub_points_indices.size()==0) return;
    if (sub_points_indices.size()==1) 
    {
        tree[index].dimension = dimension;
        tree[index].branchpoint = (*all_points)[sub_points_indices[0]][dimension];
        tree[index].point = (*all_points)[sub_points_indices[0]];
    }
    sort(sub_points_indices.begin(),sub_points_indices.end(), [&](int i,int j){return (*all_points)[i][dimension]<(*all_points)[j][dimension];} );
    //print_vector(sub_points_indices);
    //for (int i = 0 ; <)
    
    int middle_index = sub_points_indices.size()/2;
    int middle_point = sub_points_indices[middle_index];
    tree[index].dimension = dimension;
    tree[index].branchpoint = (*all_points)[sub_points_indices[middle_index]][dimension];
    tree[index].point = (*all_points)[middle_point];
    node_boundries left_boundries = boundries;
    left_boundries.limits[dimension][1] = (*all_points)[middle_index][dimension];
    left_boundries.is_set[dimension] = true;
    node_boundries right_boundries = boundries;
    right_boundries.limits[dimension][0] = (*all_points)[middle_index][dimension];
    right_boundries.is_set[dimension] = true;
    vector<int> left_points;
    vector<int> right_points;
    for(int i =0 ; i <=middle_index; i++)
    {
        left_points.push_back(sub_points_indices[i]);
    }
    for(int i =middle_index+1 ; i <sub_points_indices.size(); i++)
    {
        right_points.push_back(sub_points_indices[i]);
    }
    create_kd_tree_rec(tree, 2*index, (dimension+1)%Points_Dim ,all_points , left_points , left_boundries);
    create_kd_tree_rec(tree, 2*index+1,(dimension+1)%Points_Dim,all_points,  right_points, right_boundries);

}


void print_vector_2D (vector<vector<float>>input){
    for (int i = 0; i< input.size();i++)
    {
        for(int j = 0; j<input[0].size();j++)
        {
            cout<<input[i][j]<<" ";
        }
        cout<<endl;
    }

}


node * Create_KD_Tree(vector<vector<float>>* all_points)
{
	node * tree;
	int num_points = all_points->size();
	int tree_size = pow(2,ceil(log2(num_points)));
	int num_dimensions = (*all_points)[0].size();
	tree = new node[tree_size];
    vector<int> sub_points_indices(num_points);
    iota(sub_points_indices.begin(),sub_points_indices.end(),0); //Initializing
    node_boundries boundries;
    //shoud be implemented more efficinet
    vector<bool> temp_is_set (Points_Dim);
    vector<vector<float>> temp_limits (Points_Dim , vector<float> (2, 0));
    boundries.is_set = temp_is_set;
    boundries.limits = temp_limits;
    for (int i = 0; i<Points_Dim;i++)
    {
        boundries.is_set[i] = false;
    }
    

    create_kd_tree_rec(tree, 0, 0, all_points, sub_points_indices, boundries);
    

	
}





int main()
{
	int frame_channels = Points_Dim;
    Frame reference = read_data("0000000000.bin", Points_Dim+1, frame_channels);
    Frame query = read_data("0000000001.bin", Points_Dim+1, frame_channels);
    int num_ref_points = reference.data.size();
    int num_query_points = query.data.size();
    int num_query_points_orig = num_query_points;
    int num_points = 12	;
    cout<<pow(2,ceil(log2(num_points)));
    node * tree = Create_KD_Tree(&(reference.data));

	return 0;
}