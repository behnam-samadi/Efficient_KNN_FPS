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
#define Points_Dim 2
using namespace std;


enum dist_metric
{
    Modified_Manhattan,
    Euclidean,
    Manhattan
};
class Frame{
    public:
    int num_points;
    int points_dim;
    vector<vector<float>> data;
};

void print_vector_int (vector<int> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<v[i]<<" ";
    }
    cout<<endl;
}
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
    int children_state;
    int point_index;
};



class KD_Tree{
public:
    node * tree;
    int num_points;

void create_kd_tree_rec(node* tree, int index, int dimension, vector<vector<float>> *all_points, vector<int>sub_points_indices, node_boundries boundries)
{
 /*   if (index==5)
    {

        cout<<"oomad too";
        cout<<dimension<<" "<<endl;
        print_vector_int(sub_points_indices);
        //exit(0);
    }*/
    if (sub_points_indices.size()==0) 
        {
            return;
        }   
    if (sub_points_indices.size()==1) 
    {

        tree[index].dimension = dimension;
        tree[index].branchpoint = (*all_points)[sub_points_indices[0]][dimension];
        tree[index].point = (*all_points)[sub_points_indices[0]];
        tree[index].point_index = sub_points_indices[0];
        tree[index].boundries = boundries;
        //cout<<"with size one";
        //print_vector_float(tree[index].point);
        tree[index].is_set = 1;
        tree[index].children_state = 3;
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
    tree[index].point_index = middle_point;
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
    if ((left_points.size() > 0) && (right_points.size()>0))
    {
        tree[index].children_state = 0;
    }
    else if ((left_points.size()>0) && (right_points.size() ==0))
    {
        tree[index].children_state = 1;
    }
    else if ((left_points.size()==0) && (right_points.size()>0))
    {
        tree[index].children_state = 2;
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
int examine_point (priority_queue<pair<float, int>>* queue, int k,vector<float>reference, vector<float>query, int point_index, float max_dist)
{
    float dist = calc_distance(reference , query , Euclidean);
    if ((*queue).size() < k)
    {
        (*queue).push(make_pair(dist, point_index));
        max_dist = (*queue).top().first;
    }
    else
    {
        if (dist < max_dist)
        {
            (*queue).pop();
            (*queue).push(make_pair(dist, point_index));
            max_dist = (*queue).top().first;
        }
    }
    return max_dist;
}


vector<int> KNN_Exact_rec(vector<float> query, int k, priority_queue<pair<float, int>>* knn, int root_index)
{
    int current_node = downward_search(this->tree, query);
    int max_dist;
    bool left;
    while(current_node!=0)
    {
        max_dist = examine_point(knn, k, this->tree[current_node].point, query,this->tree[current_node].point_index , 0);
        left = ((current_node%2) == 1);
        current_node = (current_node-1)/2;
        if (left)
        {
            cross_check_cirlce_square(query , this->tree[(2*current_node+2)].boundries, max_dist);
        }
        if (right)
        {
            cross_check_cirlce_square(query , this->tree[(2*current_node+1)].boundries, max_dist);
        }

    }
}


vector<int> KNN_Exact(vector<float> query, int k)
{
    int root_index = 0;
    priority_queue<pair<float, int>> result;
    KNN_Exact_rec(query, k , &result, 0);
}


float calc_distance (vector<float> v1, vector<float> v2, dist_metric type)
{    
    if (type == Modified_Manhattan)
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
            if (type==Euclidean)
            sum+= pow(abs(v1[i] - v2[i]), 2);
            if (type==Manhattan)
            sum+= abs(v1[i] - v2[i]);
        }
        float result = sum;
        if (type == Euclidean)
            result = sqrt(result);
        return(result);
        }
}


bool line_circle_cross_check(float center_in_dimension , int line_dimension, bool up_or_down, bool is_set, float branchpoint, float radious)
{
    if (up_or_down == false) return ((center_in_dimension - radious) <= branchpoint);
    else return ((center_in_dimension + radious) >= branchpoint);
}

bool cross_check_cirlce_square(vector<float> center, node_boundries boundries, float radious)
{
    for (int d = 0; d< Points_Dim; d++)
    {
        for (int up_or_down = 0 ; up_or_down<2;up_or_down)
        {
            if (!(line_circle_cross_check(center[d], d, 1-up_or_down, boundries.is_set[d][up_or_down], boundries.limits[d][up_or_down], radious))) return false;
        }
    }
    return true;
}



int downward_search(node * tree, vector<float> query)
{
    int index = 0;
    while(!(tree[index].children_state == 3))
    {
        cout<<"point :"<<index<<endl;
        if (query[tree[index].dimension] >= tree[index].branchpoint)
            index = 2*index+2;
        else
            index = 2*index+1;
    }
    return index;
}   


    KD_Tree(vector<vector<float>>* all_points)
    {
        this->num_points = pow(2,floor(log2((*all_points).size()))+1);
        this->tree = Create_KD_Tree(all_points);
        cout<<endl<<"KD_Tree has been created with "<<this->num_points<<" points"<<endl;
    }
    ~KD_Tree()
    {
        delete[] tree;
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




int downward_search(node * tree, vector<float> query)
{
    int index = 0;
    while(!(tree[index].children_state == 3))
    {
        cout<<"point :"<<index<<endl;
        if (query[tree[index].dimension] >= tree[index].branchpoint)
            index = 2*index+2;
        else
            index = 2*index+1;
    }
    return index;
}   



bool line_circle_cross_check(float center_in_dimension , int line_dimension, bool up_or_down, bool is_set, float branchpoint, float radious)
{
    if (is_set == false) return true;
    if (up_or_down == false) return ((center_in_dimension - radious) <= branchpoint);
    else return ((center_in_dimension + radious) >= branchpoint);
}

bool cross_check_cirlce_square(vector<float> center, node_boundries boundries, float radious)
{
    for (int d = 0; d< Points_Dim; d++)
    {
        for (int up_or_down = 0 ; up_or_down<2;up_or_down++)
        {
            if (!(line_circle_cross_check(center[d], d, 1-up_or_down, boundries.is_set[d][up_or_down], boundries.limits[d][up_or_down], radious))) return false;
            //cout<<"check for dimension "<<d<<" and up_or_down "<<up_or_down<<"not finished the search"<<endl;
        }
    }
    return true;
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
    node_boundries test_boundry;
    //test_boundry.limits = {{7,8}, {9,10}, {11,12}};
    //test_boundry.is_set = {{0,0}, {0,0}, {0,0}};
    test_boundry.limits = {{5,10}, {-19,-7}};
    test_boundry.is_set = {{1,1}, {1,1}};

    cout<<endl<<cross_check_cirlce_square({7,-1}, test_boundry, 2)<<endl;
    exit(0);
    //node * tree = Create_KD_Tree(&(test_points));
    
    double runTime = -omp_get_wtime();
    //KD_Tree test_tree(&(reference.data));
    KD_Tree test_tree(&test_points);
    runTime +=omp_get_wtime();
    cout<<runTime;
    cout<<endl<<"result: "<<downward_search(test_tree.tree, {-2,7,9});
    exit(0);

    //node * tree = Create_KD_Tree(&(reference.data));
    cout<<endl<<"The KD-Tree Has been Created"<<endl;
    for (int i = 0 ; i < test_tree.num_points;i++)
    {
        
        if (test_tree.tree[i].is_set)
        {
        cout<<endl<<i<<"'th node:"<<endl;
        cout<<test_tree.tree[i].dimension<<endl;
        cout<<test_tree.tree[i].branchpoint<<endl;
        print_vector_float(test_tree.tree[i].point);
        //exit(0);
        print_vector_2D(test_tree.tree[i].boundries.limits);
        cout<<endl;
        //print_vector_2D_bool(test_tree.tree[i].boundries.is_set);
    }

    }

    
    //node * tree = Create_KD_Tree(&(reference.data));

	return 0;
}