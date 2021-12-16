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


struct thread_data
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
    cout<<"num calc: "<<calculated_distances_num<<" ";
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

void* parallel_binary_search(void * data_void)
{
    
    thread_data* data = (thread_data*)(data_void);
    Frame * reference = data->reference;
    Frame * query = data->query;
    vector<float> * query_projected = data->query_projected;
    //vector<float> * reference_projected = data->reference_projected;
    int k = data->k;
    int start_reference = data->start_reference;
    int end_reference = data->end_reference;
    int num_ref_points = data->num_ref_points;
    vector<int> * job_list = data->job_list;
    /*if (job_list->size() < 64)
    {
    cout<<"parallel_binary_search is called for"<<endl;
    cout<<"job_list "<<job_list->size();
    
    exit(0);
}*/
    pthread_mutex_t* push_mutex = data->push_mutex;
    vector<vector<int>>* result = data->result;
    vector<pthread_t*>* threads = data->threads;
    int task_size = job_list->size();
    int middle_index;
    print_vector_float((*query_projected));
    exit(0);
    vector <int> * left_list;
    vector <int>* right_list;
    vector <int>* equal_list;
    //cout<<endl<<start_reference<<endl<<end_reference<<endl;

    while(task_size>0)
    {
        middle_index = (start_reference +end_reference)/2;
        output_write<<"function called for: inja:"<<endl;
        print_vector_int(*job_list);
        left_list = new vector<int>();
        right_list = new vector<int>();
        equal_list = new vector<int>();
        for (int query_point = 0 ; query_point<task_size; query_point++)
        {
            //cout<<"in: "<<endl<<(*job_list)[query_point]<<endl;
            //cout<<"query_point: "<<query_point<<endl;
            
            if ((*query_projected)[(*job_list)[query_point]] < reference->data[middle_index][point_dim])
            {
                left_list->push_back((*job_list)[query_point]);
                //cout<<"added to left_list"<<endl;
            }
            else if ((*query_projected)[(*job_list)[query_point]] > reference->data[middle_index][point_dim])
            {
                right_list->push_back((*job_list)[query_point]);
                //cout<<"added to right_list"<<endl;
            }
            else
            { 
                equal_list->push_back((*job_list)[query_point]);
                //cout<<"added to  equal_list"<<endl;
            }
        }
        //cout<<"left_list:"<<endl;
        //print_vector_int(*left_list);
        //cout<<"right_list"<< endl;
        //print_vector_int(*right_list);
//        cout<<endl<<"equal_list"<<equal_list->size()<< endl;
        //print_vector_int(*equal_list);
        //cout<<left_list->size()<<" "<<right_list->size()<<" "<<equal_list->size()<<endl;
        
        delete  job_list;
        
        if (right_list->size() > 0)
        {
//            cout<<endl<<"oomad ke besaze"<<endl;
            


            pthread_t* temp = new pthread_t;           
            pthread_mutex_lock(push_mutex);
            threads->push_back(temp);
            pthread_mutex_unlock(push_mutex);
            thread_data* args = new thread_data;
            args->reference = reference;
            args->query_projected = query_projected;
            args->query = query;
            args->k = k;
            args->start_reference = middle_index+1;
            args->end_reference = end_reference;
//            cout<<"chizi ke ferestade shod: "<<right_list<<endl;
            args->job_list = right_list;
            args->num_ref_points = num_ref_points;
            args->result = result;
            args->threads = threads;
            args->push_mutex = push_mutex;
            pthread_create(temp, NULL, parallel_binary_search, (void*)args);
//            cout<<endl<<"eauxilary thread cretaed "<<endl;
            
        }
        if (equal_list->size() > 0)
        {
            
            for (int eq = 0 ; eq < equal_list->size(); eq++)
            {
            int nearest = binary_search((reference->data), (*query_projected)[(*equal_list)[eq]], start_reference, end_reference);
            exact_knn_projected     (result                     ,reference     ,query->row_data[(*equal_list)[eq]],(*query_projected)[(*equal_list)[eq]],nearest, k, (*equal_list)[eq],num_ref_points);    
            //void exact_knn_projected(vector<vector<int>>* output,const Frame* reference,vector<float>query, float query_projected, int nearest_index, int K, int row, int num_ref_points)

        }
    }
        if (left_list->size() > 0)
        {
//            cout<<"left_list.size() is positive";
            if (left_list->size() == 1)
            {
                int nearest = binary_search((reference->data), (*query_projected)[(*left_list)[0]], start_reference, end_reference);
                cout<<"nearest: "<<nearest<<endl;
                exact_knn_projected     (result                     ,reference     ,query->row_data[(*left_list)[0]],(*query_projected)[(*left_list)[0]],nearest, k,(*left_list)[0],num_ref_points);    
                cout<<"result: "<<endl;
                print_vector_2D(*result);
                cout<<"anjma shod"<<endl;
            }
            else
            {
                job_list = left_list;
                task_size = (*job_list).size();
                end_reference = middle_index+1;
            }
        }
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


int main()
{

    int frame_channels = 3;
    Frame reference("reformed_dataset/0_cut_20000.txt");
    Frame query("reformed_dataset/0000000001_shuffle_cut.txt");
    int num_ref_points = reference.num_points;
    int num_query_points = query.num_points;
    int num_query_points_orig = num_query_points;
    int round_size = fix_round_size;
    int round_num = num_query_points/round_size;
       vector<float> reference_projected(num_ref_points);
    vector<float> query_projected(num_query_points);
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
    
    for (int i =0 ; i<num_query_points;i++)
    {
        query_projected[i] = 0;
        for (int j = 0; j<point_dim;j++)
        {
        query_projected[i] += query.row_data[i][j];
        }
    }
    //print_float_2d(reference.data, reference.num_points, point_dim+1);
    vector<int> query_order(num_query_points);
    iota(query_order.begin(),query_order.end(),0); //Initializing
    //sort( query_order.begin(),query_order.end(), [&](int i,int j){return query_projected[i]<query_projected[j];} );
    //sort( query_projected.begin(),query_projected.end());
//print_vector_float(reference_projected);

int k = 50;
int num_threads;
vector<vector<int>> exact_fast_result  (num_query_points , vector<int> (k, 0));
vector <int> *init_joblist = new vector<int>;
for (int i = 0 ; i<round_size;i++)
{
    init_joblist->push_back(i);
}
 pthread_mutex_t* thread_mutex = new pthread_mutex_t;
 pthread_mutex_init(thread_mutex,0);


    for (int round = 0 ; round < round_num; round++)
    {
    thread_data* args = new thread_data;
    vector<pthread_t*> round_threads;
    args->reference = &reference;
    args->query = &query;
    args->job_list = init_joblist;
    //args->reference_projected = &reference_projected;
    args->k = k;
    args->query_projected = &query_projected;
    args->start_reference = 0;
    args->end_reference = num_ref_points-1;
    args->num_ref_points = num_ref_points;
    args->result = &exact_fast_result;
    args->threads = &round_threads;
    args->push_mutex = thread_mutex;
    pthread_t * main_thread = new pthread_t;
    round_threads.push_back(main_thread);
    pthread_create( main_thread, NULL, parallel_binary_search, (void*)args);
    bool round_working = true;
    for(int t = 0; t<round_threads.size();t++)
    {
        pthread_join(*(round_threads[t]),NULL);
    }
    num_threads = round_threads.size();
    for (int tn = 0; tn < round_threads.size();tn++)
    {

       delete round_threads[tn];
    }
}
exit(0);

bool cont;
cont = true;
if(cont)
{

int score = 0;
for (int q = 0 ;q <num_query_points;q++)
{
    
    vector<int> KNN_one_row_test =  KNN_one_row(&reference,&query,k,Euclidean,query_order[q]);
    int matches = 0;
    for (int j = 0; j < k; j++)
    {
        bool found = false;
        for (int c = 0 ; c<k; c++)
        {
            if (exact_fast_result[q][j] == KNN_one_row_test[c])
            {
                found = true;
                break;
            }
        }
        if (found==true)
            matches++;
    }
    
    if (matches == k)
        score++;
}
cout<<score<<" "<<num_threads<<" ";
}
}