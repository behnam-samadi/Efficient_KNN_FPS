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
#include <queue>
#define point_dim 3
#define fix_round_size 64
using namespace std;
//todo: move num_calc_dis to function
int num_calc_dis;
//todo remove


enum dist_metric
{
    Modified_Manhattan,
    Euclidean,
    Manhattan
};

class Frame{
    //later: change to private
public:
    vector<vector<float>> row_data;
    int num_points; 
    float ** data;
    Frame(string file_adress)
    {
    ifstream fin(file_adress);
    bool finished = false;
    string a1 = "start";
    int counter = 0;
    while(!finished)
    {
        //cout<<a1<<" has been read" ;
        getline(fin, a1, ',');      
        if (a1[0]!='e')
        {
            //cout<<stof(a1)<<" has been read"<<endl ;
            counter++;
            if ((counter%100) == 0)
            {
                cout<<counter<<endl;
            }
            //cout<<stof(a1)<<" has been read"<<endl ;
            row_data.push_back(vector<float>(point_dim));       
            row_data[row_data.size()-1][0] = stof(a1);
            for (int c = 1 ;c<point_dim;c++)
            {
                getline(fin, a1, ',');      
                row_data[row_data.size()-1][c] = stof(a1);
            }
        }
        else finished = true;
        num_points = row_data.size();
        
    }
    allocate_data();
    }
    void allocate_data()
    {
        //allocating 
        float * temp = new float [num_points*(point_dim+1)];
        data = new float*[num_points];
        for (int i = 0 ; i < num_points;i++)
        {
            data[i] = (temp+i*(point_dim+1));
        }
    }
};


void print_float_2d (float ** v, int X, int Y){
    for (int i = 0 ; i< X;i++)
    {
        for (int j = 0 ; j < Y ; j++)
        {

        cout<<v[i][j]<<" ";
    }
    cout<<endl;
    }
    cout<<endl;
}


void print_vector_float (vector<float> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<v[i]<<" ";
    }
    cout<<endl;
}


float calc_distance (float *v1, vector<float>v2, dist_metric type)
{
    //cout<<"distance of ";
    //print_vector_float(v1);
    //print_vector_float(v2);
    if (type == Modified_Manhattan)
    {
        float sum1 = 0;
        float sum2 = 0;
        for(int i = 0; i<point_dim;i++)
            sum1+=v1[i];
        for(int i = 0; i<point_dim;i++)
            sum2+=v2[i];
        return (abs(sum2 - sum1));
    }
    else
    {
        float sum = 0;
        for(int i = 0; i<point_dim;i++)
        {
            if (type==Euclidean)
            sum+= pow(abs(v1[i] - v2[i]), 2);
            if (type==Manhattan)
            sum+= abs(v1[i] - v2[i]);
        }
        float result = sum;
        if (type == Euclidean)
            result = sqrt(result);
        //cout<<result;
        //exit(0);
        return(result);
        }
}


vector<int> topK(vector<float> input, int K){
    float inf = std::numeric_limits<float>::max();
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
    int num_ref_points = reference->num_points;
    int num_query_points = query->num_points;
    vector<int> result(K);
    vector<float>  distance (num_ref_points);
    int i = index;
        //cout<<"KNN, Progress:" <<(float)i/num_query_points<<"\n";
        for (int j = 0; j<num_ref_points;j++)
        {
            distance[j] = calc_distance((*reference).data[j], query->row_data[i], metric);

        }
        vector<int> topk = topK(distance, K);
        //cout<<"in javabe nahayi ast:"<<endl;
        //print_vector(topk);
        
        for(int c = 0; c<K;c++)
        {
            result[c] = topk[c];
            
        }
return(result);
}


struct query_pair
{
    int index;
    float value;
};

struct root_thread_data
{
    int id_for_check;
    float middle_value;
    int * NN_result;
    vector <float> * query_points;
    bool * continue_process;
    int max_job_index, middle_index;
    queue <query_pair> * right_child_write;
    queue <query_pair> * left_child_write;
    pthread_mutex_t* right_child_write_mutex;
    pthread_mutex_t* left_child_write_mutex;    
    bool * start;
    int * num_started_threads;
    pthread_mutex_t * num_started_threads_mutex;
};

struct inner_node_thread_data
{
    int id_for_check;
    float middle_value;
    int * NN_result;
    int middle_index;
    queue <query_pair> * right_child_write;
    queue <query_pair> * left_child_write;
    queue <query_pair> * read_stream;
    pthread_mutex_t * right_child_write_mutex;
    pthread_mutex_t * left_child_write_mutex;
    pthread_mutex_t * read_stream_mutex;
    int * num_started_threads;
    pthread_mutex_t * num_started_threads_mutex;
};



struct  leaf_node_thread_data
{
    int id_for_check;
    int * NN_result;
    int start_index, end_index;
    float ** ref_data;
    queue <query_pair> * read_stream;
    pthread_mutex_t * read_stream_mutex;  
    int * num_started_threads;
    pthread_mutex_t * num_started_threads_mutex;
};


int binary_search (float ** reference, float query, int begin, int end)
{
    int length = end - begin+1;
    int end_orig = end;
    int middle_index = (begin + end) / 2;
    float middle = reference[(int)((begin + end) / 2)][point_dim];
    while (end >= begin)
    {
        middle_index = (begin + end) / 2;
        middle = reference[(int)((begin + end) / 2)][point_dim];

        if (query == middle) 
        {
            return (middle_index);
        }
        else if (query > middle) 
        {
            begin = middle_index+1;
        }
        else if(query < middle) 
            {

                end = middle_index-1;
            }
        }
        float diff1 = abs(query - middle);
        float diff2;
        float diff3;
        if (middle_index < end_orig)
        {
            diff2 = abs(query - reference[(middle_index+1)][point_dim]);
        }
        else {
            diff2 =numeric_limits<float>::max() ;
        }
        if (middle_index > 0)
        {
            diff3 = abs(query - reference[middle_index-1][point_dim]);
        }
        else
        {
            diff3 = numeric_limits<float>::max();
        }
        if ((diff1 <= diff2) && (diff1 <= diff3))  {
        return(middle_index);
        }
        else if ((diff2 <= diff1) && (diff2 <= diff3))
        {
            return(middle_index+1);
        }
        else if((diff3 <= diff2) && (diff3 <= diff1)) 
        {
        return(middle_index-1);
        }
}


struct spliting_result
{
    int divider1;
    int divider2;
};

struct spliting_state
{
    int left_size;
    int right_size;
    int middle_size;
    int divider1;
    int divider2;
    int middle_index;
};



void exact_knn_projected(vector<vector<int>>* output,const Frame* reference,vector<float>query, float query_projected, int nearest_index, int K, int row, int num_ref_points)
{
    
    int start_knn = nearest_index;
    int end_knn = nearest_index;
    while((end_knn - start_knn + 1) < K)
    {
        if (start_knn ==0)    
        {
            end_knn += (K - (end_knn - start_knn + 1));
            break;
        }
        if (end_knn == num_ref_points-1)
        {
            start_knn -= (K - (end_knn - start_knn + 1));
            break;
        }
        if ((abs((reference->data)[start_knn-1][point_dim]-query_projected)) < (abs((reference->data)[end_knn+1][point_dim]-query_projected)))
        {
            start_knn--;
        }
        else
        {
            end_knn++;
        }
    }

    float max_dist = calc_distance(reference->data[start_knn], query, Euclidean);
    num_calc_dis++;
    float dist;
    int calculated_distances_num = 0;
    priority_queue<pair<float, int>> knn;
    for(int c = start_knn; c<= end_knn; c++)
    {
        dist = calc_distance(reference->data[c], query, Euclidean);
        num_calc_dis++;

        calculated_distances_num ++;
        knn.push(make_pair(dist, c));
        if (dist > max_dist)
        {
            max_dist = dist;
        }
    }
    //cout<<" start_knn: "<<start_knn<<" end_knn: "<<end_knn<<endl;
    int right_arrow = end_knn+1;
    int left_arrow = start_knn-1;
    max_dist = knn.top().first;
    
    if (right_arrow<num_ref_points)
        {
    while( abs( reference->data[right_arrow][point_dim] - query_projected ) <= (sqrt(3)*max_dist)    )
    {
        dist = calc_distance(reference->data[right_arrow], query, Euclidean);

        num_calc_dis++;
        calculated_distances_num++;
        if (dist < max_dist)
        {
            knn.pop();
            knn.push(make_pair(dist, right_arrow));
            max_dist = knn.top().first;
        }
        right_arrow++;
        if (right_arrow == num_ref_points)
            break;
    }
}
if (left_arrow>0)
{
        while(abs(reference->data[left_arrow][point_dim] - query_projected) <= (sqrt(3)*max_dist))
    {
        dist = calc_distance(reference->data[left_arrow], query, Euclidean);
        num_calc_dis++;
        calculated_distances_num++;
        if (dist < max_dist)
        {
            
            knn.pop();
            knn.push(make_pair(dist, left_arrow));
            max_dist = knn.top().first;
        }
        left_arrow--;
        if (left_arrow<0) break;
    }
}
int c = 0;
    while(knn.size())
    {
        //cout<<endl<<"row "<<row<<"col "<<c<<"is changing";
        (*output)[row][c++] = knn.top().second;
        //cout<<endl<<"row "<<row<<"col "<<c-1<<"changed";
        knn.pop();
    }
    //cout<<"num calc: "<<calculated_distances_num<<" ";
}

void print_vector_int (vector<int> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<v[i]<<" ";
        //output_write<<endl<<v[i]<<" ";
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


void print_vector_2D_float (vector<vector<float>>input){
    for (int i = 0; i< input.size();i++)
    {
        for(int j = 0; j<input[0].size();j++)
        {
            cout<<input[i][j]<<" ";
        }
        cout<<endl;
    }

}

void * root_func (void* void_data)
{
    
    cout<<"root "<<endl;
    root_thread_data * args = (root_thread_data*)(void_data);
    float middle_value = args->middle_value;
    vector <float> * query_points = args->query_points;
    bool * continue_process = args->continue_process;
    int max_job_index = args->max_job_index;
    int * NN_result = args->NN_result;
    queue <query_pair> * right_child_write = args->right_child_write;
    queue <query_pair> * left_child_write = args->left_child_write;
    pthread_mutex_t* right_child_write_mutex = args->right_child_write_mutex;
    pthread_mutex_t* left_child_write_mutex = args->left_child_write_mutex;
    pthread_mutex_t * num_started_threads_mutex = args->num_started_threads_mutex;
    int * num_started_threads = args->num_started_threads;
    pthread_mutex_lock(num_started_threads_mutex);
    (*num_started_threads)++;
    pthread_mutex_unlock(num_started_threads_mutex);
    bool * start = args->start;
    int middle_index = args->middle_index;
    int current_query_index = 0;
    float current_query;
    query_pair temp_pair;
    while(!(*start))
    {
        //cout<<"wait"<<endl;
    }
    while(current_query_index <= max_job_index)
    {
        
        current_query = (*query_points)[current_query_index];

        //cout<<endl<<current_query_index<<" value: "<< current_query<<endl;
        temp_pair.index = current_query_index;
        temp_pair.value = current_query;
        if (current_query < middle_value)
        {

            pthread_mutex_lock(left_child_write_mutex);
            //cout<<current_query_index<<" in root"<<endl;
            left_child_write->push(temp_pair);
            //cout<<"L: "<<left_child_write->size()<<" "<<left_child_write<< endl;
            pthread_mutex_unlock(left_child_write_mutex);
        }
        else if (current_query > middle_value)
        {
            pthread_mutex_lock(right_child_write_mutex);
            right_child_write->push(temp_pair);
            //cout<<"R: "<<right_child_write->size()<<" "<<right_child_write<<endl;
            pthread_mutex_unlock(right_child_write_mutex);
        }
        else
        {
            NN_result[current_query_index] = middle_index;
        }
        current_query_index++;
    }
    temp_pair.index = -1;
    pthread_mutex_lock(left_child_write_mutex);
    left_child_write->push(temp_pair);
    pthread_mutex_unlock(left_child_write_mutex);
    pthread_mutex_lock(right_child_write_mutex);
    right_child_write->push(temp_pair);
    pthread_mutex_unlock(right_child_write_mutex);
    cout<<"end_root "<< endl;
}





void * inner_node_func (void* void_data)
{
    
    inner_node_thread_data * args = (inner_node_thread_data*)(void_data);
    int id_for_check = args->id_for_check;
    cout<<"inner "<<id_for_check<< endl;
    float middle_value = args->middle_value;
    int * NN_result = args->NN_result;
    int middle_index = args->middle_index;
    queue <query_pair> * right_child_write = args->right_child_write;
    queue <query_pair> * left_child_write = args->left_child_write;
    queue <query_pair> * read_stream = args->read_stream;
    pthread_mutex_t * num_started_threads_mutex = args->num_started_threads_mutex;
    int * num_started_threads = args->num_started_threads;
    pthread_mutex_lock(num_started_threads_mutex);
    (*num_started_threads)++;
    pthread_mutex_unlock(num_started_threads_mutex);
    pthread_mutex_t * right_child_write_mutex = args->right_child_write_mutex;
    pthread_mutex_t * left_child_write_mutex = args->left_child_write_mutex;
    pthread_mutex_t * read_stream_mutex = args->read_stream_mutex;
    bool cont = true;
    bool read;
    //int query_point;
    query_pair current_pair;
    float query_point;
    int query_index;
    while(cont)
    {

        read = false;
        pthread_mutex_lock(read_stream_mutex);
        //pthread_mutex_lock(cout_mutex);
        //cout<<" "<<id_for_check<<" read_stream->size " <<read_stream->size()<<endl;
        //pthread_mutex_unlock(cout_mutex);
        if (read_stream->size() >0)
        {
            read = true;
            current_pair = read_stream->front();
            read_stream->pop();
        }
        pthread_mutex_unlock(read_stream_mutex);
        if (read)
        {
            query_point = current_pair.value;
            query_index = current_pair.index;
            //breakpoint:
            //creating struct for pair query
            if (query_index!= -1)
            {
                if (query_point<middle_value)
                {
                    pthread_mutex_lock(left_child_write_mutex);
                    left_child_write->push(current_pair);
                    //if (id_for_check == 1) cout<<current_pair.index<<" wrote to L "<<id_for_check<<endl;
                    pthread_mutex_unlock(left_child_write_mutex);
                }
                else if (query_point > middle_value)
                {
                    pthread_mutex_lock(right_child_write_mutex);
                    right_child_write->push(current_pair);
                    //if (id_for_check == 1) cout<<current_pair.index<<" wrote to R "<<id_for_check<<endl;
                    pthread_mutex_unlock(right_child_write_mutex);
                }
                else
                {
                    (NN_result)[query_index] = query_point;
                }
            }
            else
            {
                query_pair temp_pair;
                temp_pair.index = -1;
                pthread_mutex_lock(left_child_write_mutex);
                left_child_write->push(temp_pair);
                pthread_mutex_unlock(left_child_write_mutex);
                pthread_mutex_lock(right_child_write_mutex);
                right_child_write->push(temp_pair);
                pthread_mutex_unlock(right_child_write_mutex);
                cont = false;
            }
        }

    }
    cout<<"end_inner "<<id_for_check<< endl;
}   


void * leaf_node_func (void* void_data)
{
    
    leaf_node_thread_data * args = (leaf_node_thread_data*)void_data;
    int id_for_check = args->id_for_check;
    cout<<"leaf "<<id_for_check<< endl;
    int * NN_result = args->NN_result;
    int start_index = args->start_index;
    int end_index = args->end_index;
    queue <query_pair> * read_stream = args->read_stream;
    float ** ref_data = args->ref_data;
    int * num_started_threads = args->num_started_threads;
    pthread_mutex_t * num_started_threads_mutex = args->num_started_threads_mutex;
    pthread_mutex_lock(num_started_threads_mutex);
    (*num_started_threads)++;
    pthread_mutex_unlock(num_started_threads_mutex);
    pthread_mutex_t * read_stream_mutex = args->read_stream_mutex;
    
    bool cont = true;
    bool read;
    int nearest;
    query_pair temp_pair;
    while(cont)
    {
        read = false;
        pthread_mutex_lock(read_stream_mutex);
        if (read_stream->size() > 0)
        {
            temp_pair = read_stream->front();
            read_stream->pop();
            read = true;
        }
        pthread_mutex_unlock(read_stream_mutex);
        if (read)
        {
            if (temp_pair.index != -1)
            {                
                NN_result[temp_pair.index] = binary_search(ref_data, temp_pair.value, start_index, end_index);
            }
            else
            {
                cont = false;
            }

        }

    }
    cout<<"end_leaf "<<id_for_check<< endl;
}




int main()
{
    //to do : remove
    

    //*****************************************************
    //reading frames
    //*****************************************************

    Frame reference("reformed_dataset/0_cut_20000.txt");
    Frame query("reformed_dataset/1_cut_20000.txt");
    //Frame query("reformed_dataset/0000000001_shuffle_cut.txt");
    int num_ref_points = reference.num_points;
    int num_query_points = query.num_points;
    //use defined:
    int num_queries_to_be_processesd = 16000;
    int max_job_index = num_queries_to_be_processesd -1;
    
    //*****************************************************
    //preparing the reference data
    //*****************************************************
    vector<float> reference_projected(num_ref_points);
    
    for (int i =0 ; i<num_ref_points;i++)
    {
        reference_projected[i] = 0;
        for (int j = 0; j<point_dim;j++)
        {
        reference_projected[i] += reference.row_data[i][j];
        }
    }
    vector<int> reference_order(num_ref_points);
    iota(reference_order.begin(),reference_order.end(),0); //Initializing
    sort( reference_order.begin(),reference_order.end(), [&](int i,int j){return reference_projected[i]<reference_projected[j];} );
    float sum;

    //*** sorting and creating reference.data (n+1 dims) ***
    for (int i = 0; i<num_ref_points;i++)
    {
        sum = 0;
        for (int j = 0; j<point_dim;j++)
        {
            reference.data[i][j] = reference.row_data[reference_order[i]][j];
            sum += reference.data[i][j];
        }
        reference.data[i][point_dim] = sum;
    }


    //*****************************************************
    //preparing the query data
    //*****************************************************

    vector<float> query_projected(num_query_points);
    for (int i =0 ; i<num_query_points;i++)
    {
        query_projected[i] = 0;
        for (int j = 0; j<point_dim;j++)
        {
        query_projected[i] += query.row_data[i][j];
        }
    }
    int * NN_result = new int[num_queries_to_be_processesd];
    //exit(0);

/*for (int i = 0; i <=max_job_index;i++ )
    {
                //cout<<reference.data[NN_result[i]][point_dim] <<" " <<reference.data[binary_search(reference.data, query_projected[i], 0, num_ref_points -1)][point_dim]<<endl;
                //cout<<NN_result[i] <<" " <<binary_search(reference.data, query_projected[i], 0, num_ref_points -1)<<endl;
                binary_search(reference.data, query_projected[i], 0, num_ref_points -1);
    }
    exit(0);*/
    
    

    //*****************************************************
    //preparing data for threads
    //*****************************************************
    int num_threads = 63;
    int n = log2(num_threads+1);
    bool continue_process = true;
    queue <query_pair> * data_streams = new queue <query_pair> [num_threads];
    pthread_mutex_t ** data_streams_mutex = new pthread_mutex_t * [num_threads];
    
    for(int m = 0 ; m < num_threads; m++)
    {
        data_streams_mutex[m] = new pthread_mutex_t;
        pthread_mutex_init(data_streams_mutex[m],0);
    }
    int start_reference = 0;
    int end_reference = num_ref_points - 1;

    //-----------creating arrays of start and end boundries
    int * start_boundries = new int[num_threads];
    int * end_boundries = new int[num_threads];
    int * middle_indices = new int[num_threads];
    start_boundries[0] = start_reference;
    end_boundries[0] = end_reference;
    middle_indices[0] = (start_reference+end_reference)/2;

    for (int c= 1 ; c < num_threads; c++)
    {
        if (c%2 ==0)
        {
            start_boundries[c] = (start_boundries[(c-1)/2] + end_boundries[(c-1)/2])/2 +1;
            end_boundries[c] = end_boundries[(c-1)/2];
        }
        else
        {
            start_boundries[c] = start_boundries[(c-1)/2];
            end_boundries[c] = (start_boundries[(c-1)/2] + end_boundries[(c-1)/2])/2;
        }
        middle_indices[c] = (start_boundries[c] + end_boundries[c])/2;
    }
    //for (int i = 0 ; i < num_threads; i++)
    //{
    //    cout<<start_boundries[i]<<" , "<<middle_indices[i]<<" , "<<end_boundries[i]<<endl;
    //}
    int num_started_threads;
    bool start = false;
    pthread_mutex_t * num_started_threads_mutex = new pthread_mutex_t;
    pthread_mutex_init(num_started_threads_mutex,0);

    //-----------data for root:

    root_thread_data data_for_root;
    data_for_root.query_points = &query_projected;
    data_for_root.continue_process = &continue_process;
    data_for_root.max_job_index = max_job_index;
    data_for_root.left_child_write = &(data_streams[1]);
    data_for_root.right_child_write = &(data_streams[2]);
    data_for_root.left_child_write_mutex = data_streams_mutex[1];
    data_for_root.right_child_write_mutex = data_streams_mutex[2];
    data_for_root.middle_value = reference.data[middle_indices[0]][point_dim];
    data_for_root.middle_index = middle_indices[0];
    data_for_root.NN_result = NN_result;
    data_for_root.id_for_check = 0;
    data_for_root.start = &start;
    data_for_root.num_started_threads = &num_started_threads;
    data_for_root.num_started_threads_mutex = num_started_threads_mutex;

    
    
    //----------data for inner nodes
    //----------Only the required number of data elements!. the indices will be fixed later

    inner_node_thread_data * data_for_inners = new inner_node_thread_data[(int)(pow(2,(n-1))) - 1];
    for (int i = 0 ; i < pow(2,(n-1)) - 2; i++)
    {
        data_for_inners[i].read_stream = &data_streams[i+1];
        data_for_inners[i].middle_value = reference.data[middle_indices[i+1]][point_dim];
        data_for_inners[i].middle_index = middle_indices[i+1];
        data_for_inners[i].left_child_write = &data_streams[2*(i+1)+1];
        data_for_inners[i].right_child_write = &data_streams[2*(i+1)+2];
        data_for_inners[i].read_stream_mutex = data_streams_mutex[i+1];
        data_for_inners[i].left_child_write_mutex = data_streams_mutex[2*(i+1)+1];
        data_for_inners[i].right_child_write_mutex = data_streams_mutex[2*(i+1)+2];
        data_for_inners[i].id_for_check = i+1;
        data_for_inners[i].NN_result = NN_result;
        data_for_inners[i].num_started_threads = &num_started_threads;
        data_for_inners[i].num_started_threads_mutex = num_started_threads_mutex;
    }

    //----------data for leaf nodes
    leaf_node_thread_data * data_for_leaves = new leaf_node_thread_data[(int)(pow(2,n-1))];
    
    for(int i = 0 ; i<pow(2,n-1); i++)
    {
         data_for_leaves[i].read_stream = &data_streams[i + (int)((pow(2,n-1))-1)];
         data_for_leaves[i].read_stream_mutex = data_streams_mutex[i + (int)((pow(2,n-1))-1)];
         data_for_leaves[i].start_index = start_boundries[i + (int)((pow(2,n-1))-1)];
         data_for_leaves[i].end_index = end_boundries[i + (int)((pow(2,n-1))-1)];
         data_for_leaves[i].NN_result = NN_result;
         data_for_leaves[i].ref_data = reference.data;
         data_for_leaves[i].id_for_check = i + (int)((pow(2,n-1))-1);
         data_for_leaves[i].num_started_threads = &num_started_threads;
         data_for_leaves[i].num_started_threads_mutex = num_started_threads_mutex;

    }
    
    pthread_t * threads = new pthread_t[num_threads];
    pthread_create(&(threads[0]), NULL, root_func, (void*)(&(data_for_root)));

    for (int t = 1 ; t <= pow(2,n-1)-2; t++)
    {
        
        pthread_create(&(threads[t]), NULL, inner_node_func, (void*)(&(data_for_inners[t-1])));
    }
    for (int t = pow(2, n-1)-1; t<=pow(2,n)-2;t++)
    {
        
        //pthread_create(&(threads[t]), NULL, leaf_node_func, (void*)(&(data_for_leaves[t-((int)pow(2,n-1)-1])));   
        pthread_create(&(threads[t]), NULL, leaf_node_func, (void*)(&(data_for_leaves[t - ((int)pow(2,n-1)-1)])));
    }
    while(num_started_threads<num_threads);
    exit(0);
    start = true;
    for (int t = 0 ; t<num_threads;t++)
    {
        pthread_join((threads[t]),NULL);
    }
    cout<<endl;
    exit(0);
    int score = 0;
    /*for (int i = 0; i <=max_job_index;i++ )
    {
        //cout<<i<<" "<<query_projected[i]<<endl;
     
        if (NN_result[i] ==binary_search(reference.data, query_projected[i], 0, num_ref_points -1))
        {
            score++;
        }
        else
        {
            if (reference.data[NN_result[i]][point_dim] == reference.data[binary_search(reference.data, query_projected[i], 0, num_ref_points -1)][point_dim])
                score++;
            else
            {
                //cout<<reference.data[NN_result[i]][point_dim] <<" " <<reference.data[binary_search(reference.data, query_projected[i], 0, num_ref_points -1)][point_dim]<<endl;
                cout<<NN_result[i] <<" " <<binary_search(reference.data, query_projected[i], 0, num_ref_points -1)<<endl;

            }
        }
    }
    cout<<score<<endl;*/
    
}