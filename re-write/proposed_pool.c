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
#define point_dim 3
#define fix_round_size 64
using namespace std;
//todo: move num_calc_dis to function
int num_calc_dis;
ofstream output_write("trace.txt");

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


struct job
{
    int start_index, end_index, middle_index, query_index;
    vector <float> * query_points;
};


struct thread_data
{
    Frame * reference;
    Frame * query;   
    vector<float>* query_projected;
    vector<float> * reference_projected;
    int * result;
    vector<job> * thread_created_jobs;
    int * current_query_index;
    int * num_finished_jobs;
    int * max_job_index;
    int job_size;
    bool is_orig_task;
    pthread_mutex_t* thread_created_job_mutex;
    pthread_mutex_t* orig_job_mutex;
    pthread_mutex_t* num_result_mutex;
};

struct thread_data_
{
    Frame * reference;
    Frame * query;

    vector<int> * query_order; 
    vector<float>* query_projected;
    vector<float> * reference_projected;

    int k;
    vector<vector<int>>* result;
     int start_reference;
     int end_reference;
     int start_query;
     int num_ref_points;
     int end_query;
     vector <int> *job_list;
     vector<pthread_t*>* threads;
     pthread_mutex_t *push_mutex;
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
        output_write<<endl<<v[i]<<" ";
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

void * thread_function (void *data)
{
    thread_data * args = (thread_data*)(data);
    Frame * reference = args->reference;
    Frame * query = args->query;
    vector<float>* query_projected = args->query_projected;
    vector<float> * reference_projected = args->reference_projected;
    int * result = args->result;
    vector<job> * thread_created_jobs = args->thread_created_jobs;
    int * current_query_index = args->current_query_index;
    int * num_finished_jobs = args->num_finished_jobs;
    int * max_job_index = args->max_job_index;
    int job_size = args->job_size;
    pthread_mutex_t* thread_created_job_mutex = args->thread_created_job_mutex;
    pthread_mutex_t* orig_job_mutex = args->orig_job_mutex;
    pthread_mutex_t* num_result_mutex = args->num_result_mutex;
    vector <job> * thread_created_todo;
    int start_orig_task , end_orig_task;
    bool cont = true;

    while(cont)
    {
        int orig_todo = 0;
        bool thread_created_task = false;
        bool orig_task = false;
        pthread_mutex_lock(thread_created_job_mutex);
        if (thread_created_jobs->size() > 0)
        {
            thread_created_task = true;
            thread_created_todo = thread_created_jobs->pop_front();
        }
        pthread_mutex_unlock(thread_created_job_mutex);
        if (thread_created_task == false)
        {
            pthread_mutex_lock(orig_job_mutex);
            if (is_orig_task == ture)
            {
                orig_task = true;
                start_orig_task = current_query_index;
                end_orig_task = current_query_index + job_size;
                if (end_orig_task >= max_job_index)
                {
                    end_orig_task = max_job_index;
                    is_orig_task = false;
                }
            }
            pthread_mutex_unlock(orig_job_mutex);
        }
        
    }
}


int main()
{
    //*****************************************************
    //reading frames
    //*****************************************************

    Frame reference("reformed_dataset/0_cut_20000.txt");
    Frame query("reformed_dataset/0000000001_shuffle_cut.txt");
    int num_ref_points = reference.num_points;
    int num_query_points = query.num_points;
    
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

    //*****************************************************
    //preparing common data for threads (thread job list and orig job list)
    //*****************************************************

    vector <job> thread_created_jobs;
    int current_query_index = 0,  num_finished_jobs= 0, num_threads = 64;
    int job_size = num_threads;

    //******number of query points to be propcessed:
    int max_job_index = 1280;

    int * nearest_result = new int [max_job_index];
    bool is_orig_task = true;
    pthread_mutex_t* thread_created_job_mutex = new pthread_mutex_t;
    pthread_mutex_init(thread_created_job_mutex,0);

    pthread_mutex_t* orig_job_mutex = new pthread_mutex_t;
    pthread_mutex_init(orig_job_mutex,0);

    pthread_mutex_t* num_result_mutex = new pthread_mutex_t;
    pthread_mutex_init(num_result_mutex,0);    

    pthread_t threads[num_threads];
    thread_data common_data;
    common_data.reference = &reference;
    common_data.query = &query;
    common_data.reference_projected = &reference_projected;
    common_data.query_projected = &query_projected;
    common_data.result = nearest_result;
    common_data.thread_created_jobs = &thread_created_jobs;
    common_data.current_query_index = &current_query_index;
    common_data.num_finished_jobs = &num_finished_jobs;
    common_data.max_job_index = &max_job_index;
    common_data.thread_created_job_mutex = thread_created_job_mutex;
    common_data.orig_job_mutex = orig_job_mutex;
    common_data.num_result_mutex = num_result_mutex;
    common_data.job_size = job_size;
    common_data.is_orig_task = &is_orig_task;

    for (int t = 0 ; t<num_threads;t++)
        {
            pthread_create(&(threads[t]), NULL, thread_function, (void*)(&(common_data)));
        }
    for (int t = 0; t <num_threads; t++)
        {
            pthread_join(threads[t], NULL);
        }
}