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
#define point_dim 3

#define fix_round_size 64
using namespace std;
int num = 0;
int num_exam = 0;




enum dist_metric
{
    Modified_Manhattan,
    Euclidean,
    Manhattan
};


/*
    ifstream fin(file_adress, ios::binary);
    fin.seekg(0, ios::end);
    const size_t num_elements = fin.tellg() / sizeof(double);
    fin.seekg(0, ios::beg);
    Frame frame;
    frame.points_dim = output_dims;
    frame.num_points = num_elements/points_dim;
    vector<double> data_temp(num_elements);
    vector<vector<double>> data  (num_elements/points_dim , vector<double> (output_dims, 0));
    fin.read(reinterpret_cast<char*>(&data_temp[0]), num_elements*sizeof(double));
    for (size_t i = 0; i < frame.num_points; i++){
    for(size_t j = 0; j < frame.points_dim; j++)
    {
        data[i][j] = data_temp[i*points_dim + j];
    }
    
}
    frame.data = data;
    return(frame);
*/



class Frame{
    //later: change to private
public:
    vector<vector<double>> data;
    
    //vector<vector<double>> data;
    Frame(string file_adress, int max_points = 0)
    {

    ifstream fin(file_adress, ios::binary);
    fin.seekg(0, ios::end);
    size_t num_elements = fin.tellg() / sizeof(double);
    cout<<file_adress<<file_adress<< num_elements<<endl;
    if (max_points!=0) num_elements = (max_points*Points_Dim);
    int num_points = num_elements/Points_Dim;
    fin.seekg(0, ios::beg);
    //fin.read(reinterpret_cast<char*>(&data_temp[0]), num_elements*sizeof(double))
    data = vector<vector<double>> (num_points , vector<double> (Points_Dim, 0));
    for (int c = 0 ; c<num_points; c++)
    {
        if (c%200 == 0) 
            {cout<<c<<endl;}
        fin.read(reinterpret_cast<char*>(&data[c][0]), Points_Dim*sizeof(double));
        //cout<<data[c][0]<<endl;
    }
}

};



double calc_distance_ (vector<double> v1, vector<double> v2, dist_metric type)
{    
    if (type == Modified_Manhattan)
    {
        double sum1 = 0;
        double sum2 = 0;
        for(int i = 0; i<v1.size();i++)
            sum1+=v1[i];
        for(int i = 0; i<v2.size();i++)
            sum2+=v2[i];
        return (abs(sum2 - sum1));
    }
    else
    {
        double sum = 0;
        for(int i = 0; i<v1.size();i++)
        {
            if (type==Euclidean)
            sum+= pow(abs(v1[i] - v2[i]), 2);
            if (type==Manhattan)
            sum+= abs(v1[i] - v2[i]);
        }
        double result = sum;
        if (type == Euclidean)
            result = sqrt(result);
        return(result);
        }
}
vector<int> topK_(vector<double> input, int K){
    double inf = std::numeric_limits<double>::max();
    vector<int> result(K);
    for (int c = 0; c<K; c++){
        int min_arg = 0;
        for (int j = 0; j<input.size();j++)
        {
            if(input[j] < input[min_arg]){
                min_arg = j;
            }
        }
        result[c]  = min_arg;
        input[min_arg] = inf;
    }
return (result);
}

vector<int> KNN_one_row (Frame * reference, Frame * query, int K,dist_metric metric, int index){
    int num_ref_points = (*reference).data.size();
    int num_query_points = (*query).data.size();
    vector<int> result(K);
    vector<double>  distance (num_ref_points);
    int i = index;
        //cout<<"KNN, Progress:" <<(double)i/num_query_points<<"\n";
        for (int j = 0; j<num_ref_points;j++)
        {
            distance[j] = calc_distance_((*query).data[i], (*reference).data[j], metric);
        }
        vector<int> topk = topK_(distance, K);
        //cout<<"in javabe nahayi ast:"<<endl;
        //print_vector(topk);
        
        for(int c = 0; c<K;c++)
        {
            result[c] = topk[c];
            
        }
return(result);
}

void print_vector_double (vector<double> v){
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
struct node_boundries
{
    //vector<vector<bool>> is_set (3 , vector<bool> (2, 0));
    //vector<vector<double>> limits (3 , vector<double> (2, 0));
    vector<vector<bool>> is_set;
    vector<vector<double>> limits;
};

struct node
{
    int dimension;
    double branchpoint;
    //double borders[Points_Dim][2];
    node_boundries boundries;
    vector<double> point;
    bool is_set;
    int children_state;
    int point_index;
    bool examined;
};



class KD_Tree{
public:
    node * tree;
    int num_points;

void create_kd_tree_rec(node* tree, int index, int dimension, vector<vector<double>> *all_points, vector<int>sub_points_indices, node_boundries boundries)
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
        //print_vector_double(tree[index].point);
        tree[index].is_set = 1;
        tree[index].examined = false;
        tree[index].children_state = 3;
        return;
    }
    tree[index].is_set = 1;
    tree[index].examined = false;

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
    //print_vector_double(tree[index].point);
    

        
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
        print_vector_double((*all_points)[left_points[i]]);
        cout<<endl;
    }
    cout<<endl<<"rast:"<<endl;
    for(int i = 0 ; i<right_points.size();i++)
    {
        print_vector_double((*all_points)[right_points[i]]);
    }
    */
    create_kd_tree_rec(tree, 2*index+1, (dimension+1)%Points_Dim ,all_points , left_points , left_boundries);
    create_kd_tree_rec(tree, 2*index+2,(dimension+1)%Points_Dim,all_points,  right_points, right_boundries);



}


    node * Create_KD_Tree(vector<vector<double>>* all_points)
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
    vector<vector<double>> temp_limits (Points_Dim , vector<double> (2, 0));
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
    //void create_kd_tree_rec(node* tree, int index, int dimension, vector<vector<double>> *all_points, vector<int>sub_points_indices, node_boundries boundries)
    
    return tree;
}
double examine_point (priority_queue<pair<double, int>>* queue, int k,vector<double>reference, vector<double>query, int point_index)
{
    num_exam++;
    double dist = calc_distance(reference , query , Euclidean);
    if ((*queue).size() < k)
    {
        (*queue).push(make_pair(dist, point_index));
    }
    else
    {
        if (dist==0)
        {
            //cout<<"|||||||||||||||||||||||||||||||||||||||||||||||in sefr bood:"<<endl;
            //print_vector_double(query);
            //print_vector_double(reference);
            //exit(0);
        }
        if (dist < (*queue).top().first)
        {
            //cout<<endl<<"**********************"<<endl<<dist<<"<"<<(*queue).top().first<<endl<<endl<<endl;
            (*queue).pop();
            (*queue).push(make_pair(dist, point_index));
        }
    }
    this->tree[point_index].examined = true;
    return ((*queue).top().first);
}


void KNN_Exact_rec(vector<double> query, int k, priority_queue<pair<double, int>>* knn, int root_index, double current_max_dist)
{
    double max_dist = current_max_dist;
    //if (num==50) exit(0);
    //    cout<<"Starting fucntion call for "<<root_index<<endl;
    //    cout<<"after if";
        //cout<<endl<<"examined: "<<tree[root_index].examined<<endl;
        //cout<<endl<<"examined: "<<tree[root_index].examined<<endl;
    //if (this->tree[root_index].examined == true)
    //{
    //    cout<<"terminate search because of duplication"<<endl;
    //    return;
    //}
    //else{cout<<endl<<"no terminate";}
    //cout<<"after if";  
    int leaf = downward_search(this->tree,root_index, query);
    if (leaf == root_index)
    {
     //   cout<<endl<<leaf<<"is a root index";
        max_dist = examine_point(knn, k, this->tree[leaf].point, query,this->tree[leaf].point_index);
    //    cout<<endl<<"point in position "<<leaf<<" has been examined as root and max_dist: "<<max_dist;
        return;
    }
    //if (root_index == 95962)
   // {
   //     cout<<"95962: "<<endl;
   //     cout<<current_node;
   //     exit(0);
   // }
    //cout<<"search from the root: "<<root_index<<" result: "<< leaf;
    max_dist = examine_point(knn, k, this->tree[leaf].point, query,this->tree[leaf].point_index);
    //cout<<endl<<"point in position "<<leaf<<" has been examined as root and max_dist: "<<max_dist;
    bool from_left = ((leaf%2) == 1);
    int current_node = (leaf-1)/2;
    //cout<<endl<<"go upward to : "<<current_node;
    bool root_reached = false;
    while(!root_reached)
    {
        max_dist = examine_point(knn, k, this->tree[current_node].point, query,this->tree[current_node].point_index);
      //  cout<<endl<<"point in position "<<current_node<<" has been examined";
        //cout<<endl<<"max_dist changed to: " << max_dist<<endl;
        //cout<<"vaziate priority_queue bad az pointe aval: "<<endl;
        //while((*knn).size())
       // {
        //    cout<<(*knn).top().first<<" "<<(*knn).top().second<<endl;
        //    (*knn).pop();
        //}
        //exit(0);
        
     //cout<<endl<<"from_left"<<from_left<<endl;   
        if (from_left)
        {
            //exit(0);
            if (this->tree[2*current_node+2].is_set)
                {
            if (cross_check_cirlce_square(query , this->tree[(2*current_node+2)].boundries, max_dist))
            {
                
       //         cout<<"recursive call for (L): " <<2*current_node+2<<endl;
                //exit(0);
                KNN_Exact_rec(query, k, knn, 2*current_node+2 ,max_dist);
            
            }
        }

        }
        else
        {
            if (this->tree[2*current_node+1].is_set)
        {
            if (cross_check_cirlce_square(query , this->tree[(2*current_node+1)].boundries, max_dist))
            {
                
         //       cout<<"recursive call for (R): " <<2*current_node+1<<endl;
                //exit(0);
                KNN_Exact_rec(query, k, knn, 2*current_node+1, max_dist);
            }
        }
        }
    if (current_node<=root_index) root_reached = true;
    from_left = ((current_node%2) == 1);
    current_node = (current_node-1)/2;
    //cout<<endl<<"go to parent: "<<current_node<<endl;
    }
    //cout<<"Ending fucntion call for "<<root_index<<endl<<endl;;
}


void KNN_Exact(vector<double> query, int k, int*result)
{
    //int root_index = 0;
    priority_queue<pair<double, int>> result_queue;
    KNN_Exact_rec(query, k , &result_queue, 0,0);
    int index = 0;
    while(result_queue.size())
    {
        result[index++] = result_queue.top().second;
        result_queue.pop();
    }
}


double calc_distance (vector<double> v1, vector<double> v2, dist_metric type)
{    
    if (type == Modified_Manhattan)
    {
        double sum1 = 0;
        double sum2 = 0;
        for(int i = 0; i<v1.size();i++)
            sum1+=v1[i];
        for(int i = 0; i<v2.size();i++)
            sum2+=v2[i];
        return (abs(sum2 - sum1));
    }
    else
    {
        double sum = 0;
        for(int i = 0; i<v1.size();i++)
        {
            if (type==Euclidean)
            sum+= pow(abs(v1[i] - v2[i]), 2);
            if (type==Manhattan)
            sum+= abs(v1[i] - v2[i]);
        }
        double result = sum;
        if (type == Euclidean)
            result = sqrt(result);
        return(result);
        }
}


bool line_circle_cross_check(double center_in_dimension , int line_dimension, bool up_or_down, bool is_set, double branchpoint, double radious)
{
    if (is_set == false) return true;
    if (up_or_down == false) return ((center_in_dimension - radious) <= branchpoint);
    else return ((center_in_dimension + radious) >= branchpoint);
}

bool cross_check_cirlce_square(vector<double> center, node_boundries boundries, double radious)
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


int downward_search(node * tree,int start_index, vector<double> query)
{
    int index = start_index;
    while(!(tree[index].children_state == 3))
    {
        if (tree[index].children_state == 1)
        {
            index = 2*index +1;
            break;       
        }
        if (tree[index].children_state == 2)
        {
            index = 2*index +2;
            break;       
        }
        //cout<<"point :"<<index<<endl;
        if (query[tree[index].dimension] >= tree[index].branchpoint)
        {
            index = 2*index+2;
        }
        else
        {
            index = 2*index+1;
        }
    }
    return index;
}   


    KD_Tree(vector<vector<double>>* all_points)
    {
        this->num_points = pow(2,floor(log2((*all_points).size()))+1);
        this->tree = Create_KD_Tree(all_points);
        //cout<<endl<<"KD_Tree has been created with "<<this->num_points<<" points"<<endl;
    }
    ~KD_Tree()
    {
        delete[] tree;
    }



};
struct thread_data
{
    KD_Tree * tree;
    vector<double> query;
    int * result;
    int query_index;
    int k;
};

void print_vector (vector<bool> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<v[i]<<" ";
    }
    cout<<endl;
}








void print_vector_2D (vector<vector<double>>input){
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

/*vector<int> KNN_one_row (Frame * reference, Frame * query, int K,dist_metric metric, int index){
    int num_ref_points = (*reference).data.size();
    int num_query_points = (*query).data.size();
    vector<int> result(K);
    vector<double>  distance (num_ref_points);
    int i = index;
        //cout<<"KNN, Progress:" <<(double)i/num_query_points<<"\n";
        for (int j = 0; j<num_ref_points;j++)
        {
            distance[j] = calc_distance((*query).data[i], (*reference).data[j], metric);
        }
        vector<int> topk = topK(distance, K);
        //cout<<"in javabe nahayi ast:"<<endl;
        //print_vector(topk);
        
        for(int c = 0; c<K;c++)
        {
            result[c] = topk[c];
            
        }
return(result);
}*/


/*int downward_search(node * tree, vector<double> query)
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
*/



void * KNN_KD_Tree (void* data)
{
    thread_data * args = (thread_data*) data;
    args->tree->KNN_Exact(args->query,args->k, args->result);
}




int main()
{
	int frame_channels = Points_Dim;
    
    Frame reference("reformed_dataset/0_gr.bin");
    //cout<<endl<<reference.data.size();
    
    Frame query("reformed_dataset/1_gr.bin", 64);

    //Frame query("reformed_dataset/0000000001_shuffle_cut.txt");
    int num_ref_points = reference.data.size();
    int num_query_points = query.data.size();
    //cout<<num_ref_points<<" "<<num_query_points<<endl;
    //cout<<endl<<reference.data.size()<<" "<<reference.data[0].size();
    //cout<<endl<<query.data.size()<<" "<<query.data[0].size();
    //cout<<"values:"<<endl;
    //cout<<endl<<query.data[3][0]<<endl;
    //cout<<reference.data[3][2];
    
    //cout<<endl<<num_ref_points<<" "<<num_query_points<<endl;
    int num_query_points_orig = num_query_points;
    int round_size = fix_round_size;
    int round_num = num_query_points/round_size;
    KD_Tree reference_tree(&(reference.data));
    exit(0);
    int num_temp_tets = 64;
    int ** result_temp = new int *[num_temp_tets];
    int test_k = 1;
    int num_exam_sum;
    double time_sum = 0;
    double time_exact_sum = 0;
    double runTime2;
    for (int q= 0 ; q< num_temp_tets;q++)
    {
        result_temp[q] = new int[test_k];
        reference_tree.KNN_Exact(query.data[q],test_k, result_temp[q]);       
    }

}