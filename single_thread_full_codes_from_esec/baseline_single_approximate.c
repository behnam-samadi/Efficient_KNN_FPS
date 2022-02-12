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

int sum_size_of_buckets;


enum dist_metric
{
    Modified_Manhattan,
    Euclidean,
    Manhattan
};







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
    vector<double> point;
    node * left_child;
    node * right_child;
    double * bucket;
    int size_of_bucket;
    int * bucket_indices;
    bool is_set;
    int children_state;
    int point_index;
    bool examined;
};


class KD_Tree{
public:
    node * tree;
    int num_points;
void create_kd_tree_rec(node* tree, int dimension, vector<vector<double>> *all_points, vector<int>sub_points_indices, int bucket_size)
{
    cout<<"oomad too rec"<<endl;
    if (sub_points_indices.size() <= bucket_size)
    {
        tree->bucket = new double[sub_points_indices.size()* Points_Dim];
        tree->bucket_indices = new int[sub_points_indices.size()];
        for (int p_num = 0 ; p_num<sub_points_indices.size();p_num++)
        {
            tree->bucket_indices[p_num] = sub_points_indices[p_num];
            for (int p_dim = 0 ; p_dim<Points_Dim; p_dim++)
            {
                tree->bucket[p_num*Points_Dim + p_dim] = (*all_points)[p_num][p_dim];
            }
        }
        tree->size_of_bucket = sub_points_indices.size();
        sum_size_of_buckets+=tree->size_of_bucket;
    }
    else
    {
    cout<<"oomad to else: "<<endl;
    tree->bucket = NULL;
    
    //tree[index].examined = false;
    sort(sub_points_indices.begin(),sub_points_indices.end(), [&](int i,int j){return (*all_points)[i][dimension]<(*all_points)[j][dimension];} );
    int middle_index = (sub_points_indices.size()-1)/2;
    int middle_point = sub_points_indices[middle_index];
    cout<<"middle_index :"<<middle_index<<endl;
    tree->dimension = dimension;
    tree->branchpoint = (*all_points)[sub_points_indices[middle_index]][dimension];
    //tree->point = (*all_points)[middle_point];
    //tree->point_index = middle_point;
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
    if (left_points.size()>0)
    {
        tree->left_child = new node;
        create_kd_tree_rec(tree->left_child, (dimension+1)%Points_Dim ,all_points , left_points , bucket_size);
    }
    else
    {
        tree->left_child = NULL;
    }
    if (right_points.size()>0)
    {
        tree->right_child = new node;
        create_kd_tree_rec(tree->right_child,(dimension+1)%Points_Dim,all_points,  right_points,  bucket_size);
    }
    else
    {
        tree->right_child = NULL;
    }   
    }

}
    node * Create_KD_Tree(vector<vector<double>>* all_points, int bucket_size)
{
    node * tree;
    int num_dimensions = (*all_points)[0].size();
    tree = new node;
    num_points = all_points->size();
    cout<<"before vector"<<endl;
    cout<<"num_points "<<num_points;
    
    vector<int> sub_points_indices(num_points);
    cout<<"after vector"<<endl;
    
    iota(sub_points_indices.begin(),sub_points_indices.end(),0); //Initializing
    create_kd_tree_rec(tree, 0, all_points, sub_points_indices, bucket_size);
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


/*void KNN_Exact_rec(vector<double> query, int k, priority_queue<pair<double, int>>* knn, int root_index, double current_max_dist)
{
    double max_dist = current_max_dist;
    int leaf = downward_search(this->tree,root_index, query);
    if (leaf == root_index)
    {
        max_dist = examine_point(knn, k, this->tree[leaf].point, query,this->tree[leaf].point_index);
        return;
    }
    max_dist = examine_point(knn, k, this->tree[leaf].point, query,this->tree[leaf].point_index);
    bool from_left = ((leaf%2) == 1);
    int current_node = (leaf-1)/2;
    bool root_reached = false;
    while(!root_reached)
    {
        max_dist = examine_point(knn, k, this->tree[current_node].point, query,this->tree[current_node].point_index);
        if (from_left)
        {
            if (this->tree[2*current_node+2].is_set)
                {
            if (cross_check_cirlce_square(query , this->tree[(2*current_node+2)].boundries, max_dist))
            {
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
                KNN_Exact_rec(query, k, knn, 2*current_node+1, max_dist);
            }
        }
        }
    if (current_node<=root_index) root_reached = true;
    from_left = ((current_node%2) == 1);
    current_node = (current_node-1)/2;
    }
}


void KNN_Exact(vector<double> query, int k, int*result)
{
    priority_queue<pair<double, int>> result_queue;
    KNN_Exact_rec(query, k , &result_queue, 0,0);
    int index = 0;
    while(result_queue.size())
    {
        result[index++] = result_queue.top().second;
        result_queue.pop();
    }
}

*/
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




double calc_distance_vec_adress (vector<double> v1, double * v2, dist_metric type)
{    
    if (type == Modified_Manhattan)
    {
        double sum1 = 0;
        double sum2 = 0;
        for(int i = 0; i<v1.size();i++)
            sum1+=v1[i];
        for(int i = 0; i<v1.size();i++)
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


vector <int> downward_search(node * tree,vector<double> query,vector<vector<double>>* all_points , int k)
{
    vector<int> result;
    while(!(tree->bucket))
    {
        if (query[tree->dimension] > tree->branchpoint)
        {
            tree = tree->right_child;
        }
        else
        {
            tree = tree->left_child;
        }
    }
    priority_queue<pair<double, int>> queue;
    //for (int i = 0 ; i < tree->size_of_bucket; i++)
    //{
    //    cout<<tree->bucket_indices[i]<<" ";
    //}
    //cout<<endl;
    double dist;
//    double max_dist = calc_distance_vec_adress(query, tree->bucket[i*Points_Dim]);
    //double dist;
    //queue.push_back(max_dist);

    for (int i = 0 ; i <  k ; i++)
    {
        //dist = calc_distance_vec_adress(query, &(tree->bucket[i*Points_Dim]), Euclidean);
        dist = calc_distance(query , (*all_points)[tree->bucket_indices[i]], Euclidean);
        //cout<<"dist: "<<dist<<endl;
        queue.push( make_pair(dist,i) );
        //double calc_distance_vec_adress (vector<double> v1, double * v2, dist_metric type)
    }
    double max_dist = queue.top().first;
    for (int i = k ; i<tree->size_of_bucket; i++)
    {
        //dist = calc_distance_vec_adress(query, (&(tree->bucket[i*Points_Dim])), Euclidean);
        //cout<<"the distance beetween: "<<endl;
        //print_vector_double(query);
        //cout<<"and: "<<endl;
        //print_vector_double((*all_points)[tree->bucket_indices[i]]);
        dist = calc_distance(query , (*all_points)[tree->bucket_indices[i]], Euclidean);
        //cout<<"dist: "<<dist<<endl;
        if (dist < max_dist)
        {
            queue.pop();
            queue.push(make_pair(dist, i));
            max_dist = queue.top().first;
        }
    }
    

    while(queue.size())
    {
        //cout<<"final result:---------------------"<<endl;
        //cout<<"----------------------------------"<<endl;
        //cout<<queue.top().first<<" "<<queue.top().second<<endl;
        result.push_back(tree->bucket_indices[queue.top().second]);
        //cout<<"----------------------------------"<<endl;
        //cout<<"----------------------------------"<<endl;
        queue.pop();
    }
    /*while(!(tree[index].children_state == 3))
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
        if (query[tree[index].dimension] >= tree[index].branchpoint)
        {
            index = 2*index+2;
        }
        else
        {
            index = 2*index+1;
        }
    }
    return index;*/
    return result;
}   


    KD_Tree(vector<vector<double>>* all_points, int bucket_size)
    {
        this->tree = Create_KD_Tree(all_points, bucket_size);
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


void print_vector_2D_bool (vector<vector<bool>>input)
{
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
    //args->tree->KNN_Exact(args->query,args->k, args->result);
}




int main()
{
	int frame_channels = Points_Dim;
    Frame reference("reformed_dataset/0_gr.bin");
    Frame query("reformed_dataset/1_gr.bin", 1280);
    int num_ref_points = reference.data.size();
    int num_query_points = query.data.size();
    int num_query_points_orig = num_query_points;
    int round_size = fix_round_size;
    int round_num = num_query_points/round_size;
    int bucket_size = 512;
    KD_Tree reference_tree(&(reference.data), bucket_size);
    int matches = 0;
    cout<<endl<<sum_size_of_buckets<<endl;
    cout<<num_ref_points;
    
    cout<<"tree created";
    int test_k = 5;
    cout<<"az inja:"<<endl;
    vector<int> baseline_result;
    vector<int> ground_truth;
    int sum_matches = 0;

    for (int q = 0 ; q < num_query_points; q++)
    {
    baseline_result = reference_tree.downward_search(reference_tree.tree,query.data[q],&(reference.data), test_k);
    ground_truth = KNN_one_row (&reference, &query, test_k,Euclidean, q);
    matches = 0;
    for (int i = 0 ; i<test_k;i++)
    {
        for (int j =0 ; j<test_k;j++)
        {
            if (baseline_result[i] == ground_truth[j])
            {
                matches++;
                break;
            }
        }
    }
    sum_matches+=matches;
}
    cout<<sum_matches<<endl<< ((float)sum_matches / (float)num_query_points)/(float)test_k*100 <<'%'<< " accuracy"<<endl;;

    //print_vector_int(reference_tree.downward_search(reference_tree.tree,query.data[21],&(reference.data), test_k));
    //print_vector_int(KNN_one_row (&reference, &query, test_k,Euclidean, 21));   

    exit(0);
    int num_temp_tets = 64;
    int ** result_temp = new int *[num_temp_tets];
    //int test_k = 1;
    int num_exam_sum;
    double time_sum = 0;
    double time_exact_sum = 0;
    double runTime2;
    for (int q= 0 ; q< num_temp_tets;q++)
    {
        result_temp[q] = new int[test_k];
        //reference_tree.KNN_Exact(query.data[q],test_k, result_temp[q]);       
    }

}