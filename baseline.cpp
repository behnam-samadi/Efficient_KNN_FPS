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
    vector<vector<bool>> is_set;
    vector<vector<float>> limits;
};

struct node
{
    int dimension;
    float branchpoint;
    //float borders[Points_Dim][2];
    node_boundries boundries;
    vector<float> point;
    bool is_set;
};



class KD_Tree{
public:
    node * tree;
    int num_points;

void create_kd_tree_rec(node* tree, int index, int dimension, vector<vector<float>> *all_points, vector<int>sub_points_indices, node_boundries boundries)
{
    if (sub_points_indices.size()==0) 
        {
            return;
        }   
    if (sub_points_indices.size()==1) 
    {

        tree[index].dimension = dimension;
        tree[index].branchpoint = (*all_points)[sub_points_indices[0]][dimension];
        tree[index].point = (*all_points)[sub_points_indices[0]];
        tree[index].boundries = boundries;
        //cout<<"with size one";
        //print_vector_float(tree[index].point);
        tree[index].is_set = 1;
        return;
    }
tree[index].is_set = 1;
//cout<<endl<<"sorted:"<<endl;
    sort(sub_points_indices.begin(),sub_points_indices.end(), [&](int i,int j){return (*all_points)[i][dimension]<(*all_points)[j][dimension];} );
    //print_vector_int(sub_points_indices);
    //for (int i = 0 ; <)
    
    int middle_index = (sub_points_indices.size()-1)/2;
    int middle_point = sub_points_indices[middle_index];
    tree[index].dimension = dimension;
    tree[index].branchpoint = (*all_points)[sub_points_indices[middle_index]][dimension];
    tree[index].point = (*all_points)[middle_point];
    //cout<<"az inja:"<<endl;
    //cout<<endl<<tree[index].dimension<<endl;
    //cout<<endl<<tree[index].branchpoint<<endl;
    //cout<<"point for "<<index<<"'th point: "<<endl;
    //print_vector_float(tree[index].point);
    

        
    node_boundries left_boundries = boundries;
    left_boundries.limits[dimension][1] = (*all_points)[middle_point][dimension];
    left_boundries.is_set[dimension][1] = true;
    node_boundries right_boundries = boundries;
    right_boundries.limits[dimension][0] = (*all_points)[middle_point][dimension];
    right_boundries.is_set[dimension][0] = true;
    tree[index].boundries = boundries;
    //cout<<"check kardane boundries:"<<endl;
    //print_vector_2D_bool(left_boundries.is_set);
    //print_vector_2D_bool(right_boundries.is_set);
    //cout<<endl;
    //print_vector_2D(left_boundries.limits);
    //print_vector_2D(right_boundries.limits);

    //exit(0);


    vector<int> left_points;
    vector<int> right_points;
    for(int i =0 ; i <middle_index; i++)
    {
        left_points.push_back(sub_points_indices[i]);
    }
    for(int i =middle_index+1 ; i <sub_points_indices.size(); i++)
    {
        right_points.push_back(sub_points_indices[i]);
    }
    
    /*
    cout<<endl<<"chapo rast:"<<endl;
    cout<<"chap"<<endl;
    for(int i = 0 ; i<left_points.size();i++)
    {
        print_vector_float((*all_points)[left_points[i]]);
        cout<<endl;
    }
    cout<<endl<<"rast:"<<endl;
    for(int i = 0 ; i<right_points.size();i++)
    {
        print_vector_float((*all_points)[right_points[i]]);
    }
    */
    create_kd_tree_rec(tree, 2*index+1, (dimension+1)%Points_Dim ,all_points , left_points , left_boundries);
    create_kd_tree_rec(tree, 2*index+2,(dimension+1)%Points_Dim,all_points,  right_points, right_boundries);



}


    node * Create_KD_Tree(vector<vector<float>>* all_points)
{
    node * tree;
    int num_points = all_points->size();
    int tree_size = pow(2,floor(log2(num_points))+1);
    int num_dimensions = (*all_points)[0].size();
    tree = new node[tree_size];
    vector<int> sub_points_indices(num_points);
    iota(sub_points_indices.begin(),sub_points_indices.end(),0); //Initializing
    node_boundries boundries;
    //shoud be implemented more efficinet
    vector<vector<bool>> temp_is_set (Points_Dim , vector<bool> (2, 0));
    vector<vector<float>> temp_limits (Points_Dim , vector<float> (2, 0));
    boundries.is_set = temp_is_set;
    boundries.limits = temp_limits;
    for (int i = 0; i<Points_Dim;i++)
    {
        for (int j = 0 ; j <Points_Dim;j++)
        {
            boundries.is_set[i][j] = false;
        }
    }
    create_kd_tree_rec(tree, 0, 0, all_points, sub_points_indices, boundries);
    cout<<endl<<endl<<endl<<"kdtree cretaed";
    return tree;
}

    KD_Tree(vector<vector<float>>* all_points)
    {
        this->num_points = (*all_points).size();
        this->tree = Create_KD_Tree(all_points);
    }
    ~KD_Tree()
    {
        delete tree;
    }

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

void print_vector_int (vector<int> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<v[i]<<" ";
    }
    cout<<endl;
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


void print_vector_2D_bool (vector<vector<bool>>input){
    for (int i = 0; i< input.size();i++)
    {
        for(int j = 0; j<input[0].size();j++)
        {
            cout<<input[i][j]<<" ";
        }
        cout<<endl;
    }

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
    vector<vector<float>> test_points = {{2,3,7}, {4,-1,5}, {7,-1,0}, {0,0,0}, {1,2,6}, {0,5,-5}, {-2,7,9}, {5,0,0,}};
    //node * tree = Create_KD_Tree(&(test_points));
    KD_Tree test_tree(&test_points);

    //node * tree = Create_KD_Tree(&(reference.data));
    cout<<endl<<"The KD-Tree Has been Created"<<endl;
    for (int i = 0 ; i < test_tree.num_points;i++)
    {
        if (test_tree.tree[i].is_set)
        {
        cout<<endl<<i<<"'th node:"<<endl;
        //cout<<tree[i].dimension<<endl;
        //cout<<tree[i].branchpoint<<endl;
        //print_vector_float(tree[i].point);
        //exit(0);
        print_vector_2D(test_tree.tree[i].boundries.limits);
        cout<<endl;
        print_vector_2D_bool(test_tree.tree[i].boundries.is_set);
    }

    }

    
    //node * tree = Create_KD_Tree(&(reference.data));

	return 0;
}