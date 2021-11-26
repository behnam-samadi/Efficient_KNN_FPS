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
#define fix_round_size 512
using namespace std;
//todo: move num_calc_dis to function
int num_calc_dis;

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
    cout<<"start reading file "<<file_adress<<endl;
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


    int k;
    vector<vector<int>>* result;
     int start_reference;
     int end_reference;
     int start_query;
     int num_ref_points;
     int end_query;
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

spliting_result binary_search_split(vector<float> *input, int start_index, int end_index, float query)
{

    int start_orig = start_index;
    int end_orig = end_index;
    bool successful = 0;
    int middle_index;
    while (end_index > start_index)
    {
        middle_index = (end_index + start_index) / 2;
        if (query > (*input)[middle_index])
        {
            start_index = middle_index + 1;
        }
        else if (query < (*input)[middle_index])
        {
            end_index = middle_index - 1;
        }
        else if(query == (*input)[middle_index])
        {
            successful = 1;
            break;
        }       
    }
    if(start_index == end_index)
    {
        middle_index = (end_index + start_index) / 2;
        if(query == (*input)[middle_index])
        {
            successful = 1;
        }       
    }
    spliting_result result;
    if (!successful)
    {
        int divide_point = start_index;
        if ( ((*input)[divide_point] < query)) 
            {divide_point++;}

        if (end_index == -1) divide_point = 0;
        result.divider1 = divide_point;
        result.divider2 = divide_point;
    }
    else
    {
        int divide_point1 = middle_index;
        int divide_point2 = middle_index;
        while(divide_point1 > 0) 
        {
         if ((*input)[divide_point1-1]==query) divide_point1--;
         else break;
        }
        while(divide_point2 < end_index)
        {
            if ((*input)[divide_point2+1]==query) divide_point2++;
            else break;
        }
        divide_point2++;
        result.divider1 = divide_point1;
        result.divider2 = divide_point2;
    }

    return result;
}

spliting_result binary_search_split_(vector<vector<float>> *input, int start_index, int end_index, float query)
{

    int start_orig = start_index;
    int end_orig = end_index;
    bool successful = 0;
    int middle_index;
    while (end_index > start_index)
    {
        middle_index = (end_index + start_index) / 2;
        if (query > (*input)[middle_index][point_dim])
        {
            start_index = middle_index + 1;
        }
        else if (query < (*input)[middle_index][point_dim])
        {
            end_index = middle_index - 1;
        }
        else if(query == (*input)[middle_index][point_dim])
        {
            successful = 1;
            break;
        }       
    }
    if(start_index == end_index)
    {
        middle_index = (end_index + start_index) / 2;
        if(query == (*input)[middle_index][point_dim])
        {
            successful = 1;
        }       
    }
    spliting_result result;
    if (!successful)
    {
        int divide_point = start_index;
        if ( ((*input)[divide_point][point_dim] < query)) 
            {divide_point++;}

        if (end_index == -1) divide_point = 0;
        result.divider1 = divide_point;
        result.divider2 = divide_point;
    }
    else
    {
        int divide_point1 = middle_index;
        int divide_point2 = middle_index;
        while(divide_point1 > 0) 
        {
         if ((*input)[divide_point1-1][point_dim]==query) divide_point1--;
         else break;
        }
        while(divide_point2 < end_index)
        {
            if ((*input)[divide_point2+1][point_dim]==query) divide_point2++;
            else break;
        }
        divide_point2++;
        result.divider1 = divide_point1;
        result.divider2 = divide_point2;
    }

    return result;
}



void exact_knn_projected(vector<vector<int>>* output,const Frame* reference,vector<float>query, float query_projected, int nearest_index, int K, int row, int num_ref_points)
{
    cout<<"exact search is called for "<<query_projected<<" "<<nearest_index<<endl;
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
    cout<<" start_knn: "<<start_knn<<" end_knn: "<<end_knn<<endl;
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
}

spliting_state one_step_parallel_binary_search(vector<float>* query_projected,int start_reference, int end_reference, int start_query, int end_query, vector<vector<int>>* result, Frame* reference, Frame* query, vector<int>* query_order, int k, int num_ref_points)
{
    
    int middle_index = (start_reference + end_reference)/2;
    float middle_value = reference->data[middle_index][point_dim];
    spliting_result split = binary_search_split(query_projected, start_query, end_query, middle_value);
    int divider1 = split.divider1;
    int divider2 = split.divider2;
    for(int d = divider1; d< divider2;d++)
    {        
     exact_knn_projected     (result,reference,query->row_data[(*query_order)[d]],(*query_projected)[d],middle_index, k,d,num_ref_points);    

    }

    spliting_state state;
    state.left_size = max(divider1 - start_query, 0);
    state.middle_size = max(divider2 - divider1, 0);
    state.right_size = max(end_query - divider2 + 1, 0);
    state.divider1 = divider1;
    state.divider2 = divider2;
    state.middle_index = middle_index;
    return state;
}


void* parallel_binary_search(void * data_void)
{
    

    thread_data* data = (thread_data*)(data_void);
    Frame * reference = data->reference;
    Frame * query = data->query;
    vector<float> * query_projected = data->query_projected;
    vector<int> * query_order = data->query_order;
    int k = data->k;
    int start_reference = data->start_reference;
    int end_reference = data->end_reference;
    int start_query = data->start_query;
    int end_query = data->end_query;
    int num_ref_points = data->num_ref_points;
    pthread_mutex_t* push_mutex = data->push_mutex;
    vector<vector<int>>* result = data->result;
    vector<pthread_t*>* threads = data->threads;
    int middle_index;
    //cout<<endl<<"start_query: "<<start_query<<" end_query: "<<end_query;
    spliting_state state;
    do{   
    if ((end_reference - start_reference)<2)
    {
        for (int q = start_query ; q<= end_query;q++)
        {
            int nearest = binary_search((reference->data), (*query_projected)[q], start_reference, end_reference);
            //int binary_search (vector<vector<float>>* reference, float query, int begin, int end)
            //cout<<endl<<"Line:508 : q: "<<q;
         exact_knn_projected     (result                     ,reference,query->row_data[(*query_order)[q]],(*query_projected)[q],nearest, k,q,num_ref_points);       
         //void exact_knn_projected(vector<vector<int>>* output,const Frame* reference,vector<float>query, float query_projected, int nearest_index, int K, int row, int num_ref_points)
        }
        delete data;
        return NULL;
    }
    state = one_step_parallel_binary_search(query_projected , start_reference, end_reference , start_query, end_query, result, reference, query, query_order, k, num_ref_points);
    middle_index = state.middle_index;
        if(state.right_size > 0)
        {

            pthread_t* temp = new pthread_t;           
            pthread_mutex_lock(push_mutex);
            threads->push_back(temp);
            pthread_mutex_unlock(push_mutex);
            thread_data* args = new thread_data;
            args->reference = reference;
            args->query_projected = query_projected;
            args->query = query;
            args->query_order = query_order;
            args->k = k;
            args->start_reference = middle_index;
            args->end_reference = end_reference;
            args->start_query = state.divider2;
            args->num_ref_points = num_ref_points;
            args->end_query = end_query;
            //cout<<endl<<" args->start_query "<<args->start_query<<" args->end_query "<<args->end_query;
            args->result = result;
            args->threads = threads;
            args->push_mutex = push_mutex;
            if (args->end_query >= args->start_query)
            {
                //cout<<"start creting a thread"<<endl;
                pthread_create(temp, NULL, parallel_binary_search, (void*)args);
            }
            else
            {
                delete args;
            }
        }

        end_query = state.divider1-1;
        end_reference = middle_index;
    } while(state.left_size>1);
    if (state.left_size == 1)
        {
            int nearest = binary_search((reference->data), (*query_projected)[start_query], start_reference, end_reference);
            //cout<<endl<<"Line:555 : start_query: "<<start_query;
            exact_knn_projected     (result                     ,reference     ,query->row_data[(*query_order)[start_query]],(*query_projected)[start_query],nearest, k,start_query,num_ref_points);    
        }
        delete data;
        
}

void print_vector_int (vector<int> v){
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
    cout<<reference.num_points<<endl;
    cout<<query.num_points<<endl;
    //print_vector_float(query.data[1]);
    //cout<<query.data.size();
    int num_ref_points = reference.num_points;
    int num_query_points = query.num_points;
    cout<<num_ref_points<<endl<<num_query_points<<endl;
    
    

    int num_query_points_orig = num_query_points;
    int round_size = fix_round_size;
    int round_num = num_query_points/round_size;
    
       vector<float> reference_projected(num_ref_points);
    vector<float> query_projected(num_query_points);

    for (int i =0 ; i<num_ref_points;i++)
    {
        //cout<<i<<endl;
        reference_projected[i] = 0;
        for (int j = 0; j<point_dim;j++)
        {
        reference_projected[i] += reference.row_data[i][j];
        }
    }
    //print_vector_float(reference_projected);
    
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
    //        if (i < 90)cout<<"transfomration for "<<i<<" "<<j<<"'th is changed to "<<reference.row_data[reference_order[i]][j]<<endl;
            sum += reference.data[i][j];
        }
        reference.data[i][point_dim] = sum;
        //reference.data[i][point_dim] = reference_projected[i];
    //    if (i<90) cout<<"and sum is "<<reference.data[i][point_dim]<<endl;
    }
    
    for (int i =0 ; i<num_query_points;i++)
    {
        query_projected[i] = 0;
        for (int j = 0; j<point_dim;j++)
        {
        query_projected[i] += query.row_data[i][j];
        }
    }
cout<<"inja:";

    cout<<endl;


    
    cout<<reference.data[0][0];


    print_float_2d(reference.data, reference.num_points, point_dim+1);
    
    
    //cout<<"projection done!"<<endl;

    

    vector<int> query_order(num_query_points);
    iota(query_order.begin(),query_order.end(),0); //Initializing
    //sort( query_order.begin(),query_order.end(), [&](int i,int j){return query_projected[i]<query_projected[j];} );
    //sort( query_projected.begin(),query_projected.end());





    int K_test = 50;
    int num_temp_tets = 64;
    vector<vector<int>> result_test  (num_temp_tets , vector<int> (K_test, 0));
    int score = 0;
    int num_calc_dis_test = 0;
    double avg_test_time = 0;
    double avg_ex_time = 0;
    double test_time;
    vector<int> KNN_one_row_test;
    for (int q= 0 ; q< num_temp_tets;q++)
    {
        cout<<q<<endl;
        num_calc_dis = 0;
        cout<<endl<<"test_number: "<<q<<endl;
        test_time = -omp_get_wtime();
        int nearest_index = binary_search (reference.data,query_projected[q], 0, num_ref_points);

        exact_knn_projected(&result_test,&reference,query.row_data[query_order[q]],query_projected[q], nearest_index, K_test, q,num_ref_points);
        //void exact_knn_projected(vector<vector<int>>* output,const Frame* reference,float*query, float query_projected, int nearest_index, int K, int row, int num_ref_points)
        test_time+=omp_get_wtime();
        avg_test_time+= test_time;
        num_calc_dis_test+=num_calc_dis;
        //cout<<num_calc_dis<<endl;
        test_time=-omp_get_wtime();
        KNN_one_row_test =  KNN_one_row(&reference,&query,K_test,Euclidean,query_order[q]);
        test_time+=omp_get_wtime();
        avg_ex_time+= test_time;
        int matches = 0;
        for (int j = 0; j < K_test; j++)
        {
            bool found = false;
            for (int c = 0 ; c<K_test; c++)
            {
                if (result_test[q][j] == KNN_one_row_test[c])
                {
                    found = true;
                    break;
                }
            }
            if (found==true)
                matches++;
        }
        
        if (matches == K_test)
            {score++;}
        else
            {cout<<"unsuceful sraech in "<<endl<<q<<endl;
        cout<<endl<<"matches:"<<matches<<endl;
//cout<<"-----------------------------------------------result:"<<endl;
//print_vector_int(result_test[q]);
//cout<<"-----------------------------------------------exact:"<<endl;
        //print_vector_int(KNN_one_row_test);
        //cout<<"-----------------------------------------------"<<endl;
        
    }

            


        //inja

    }
    //print_vector_2D(result_test);
    avg_test_time/= num_temp_tets;
    avg_ex_time/= num_temp_tets;
    cout<<endl<<" test done with score "<<score<<endl<<" rate of cutoff: "<<(num_calc_dis_test/num_temp_tets)/(float)(num_ref_points)<<endl<<"number of calculated distances: "<<num_calc_dis_test<<endl<<"average euc num: "<<num_calc_dis_test/num_temp_tets<<endl<<" num tests: "<<num_temp_tets<<endl<<"avg test time: "<<avg_test_time<<endl<<"avg_ex_time: "<<avg_ex_time<<endl<<"speedup: "<<avg_ex_time/avg_test_time<<endl<<" "<<1/((num_calc_dis_test/num_temp_tets)/(float)(num_ref_points)) <<endl;
    



exit(0);







int k = 50;
int num_threads;
vector<vector<int>> exact_fast_result  (num_query_points , vector<int> (k, 0));
double runTime = -omp_get_wtime();
    for (int round = 0 ; round < round_num; round++)
    {
    thread_data* args = new thread_data;
    vector<pthread_t*> round_threads;
    args->reference = &reference;
    args->query = &query;
    //args->reference_projected = &reference_projected;
    args->k = k;
    args->query_order = &query_order;
    args->query_projected = &query_projected;
    args->start_reference = 0;
    args->end_reference = num_ref_points-1;
    args->num_ref_points = num_ref_points;
    args->start_query = round*round_size;
    args->end_query = (round+1)*round_size-1;
    args->result = &exact_fast_result;
    args->threads = &round_threads;
    //args->reference_order = &reference_order;
    args->push_mutex = new pthread_mutex_t;
    pthread_t * main_thread = new pthread_t;
    round_threads.push_back(main_thread);
    pthread_mutex_init(args->push_mutex,0);
    
    pthread_create( main_thread, NULL, parallel_binary_search, (void*)args);

    bool round_working = true;
    for(int t = 0; t<round_threads.size();t++)
    {
        //cout<<"started for "<<args->start_query<<" "<<args->end_query<<" to "<<t<<" "<<"from "<<round_threads.size()<<" "<<endl;
        pthread_join(*(round_threads[t]),NULL);
    }

    num_threads = round_threads.size();
    for (int tn = 0; tn < round_threads.size();tn++)
    {

       delete round_threads[tn];
    }
}
runTime +=omp_get_wtime();
cout<<endl<<runTime<<endl;



cout<<"calculation time: "<<runTime<<endl<<"Do you want to perform an accuracy check? (1or0)"<<endl;
bool cont;
cin>>cont;
if(cont)
{

int score = 0;
for (int q = 0 ;q <num_query_points;q++)
{
    cout<<endl<<"test number: "<<q<<" from "<<num_query_points;
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
cout<<endl;
cout<<"Number of Query Points:         "<<num_query_points<<endl;
cout<<"Number of Nearest Neighbors (K):"<<k<<endl;
cout<<"Number of Correct Answers:      "<<score<<endl;
cout<<"Execution Time:                 "<<runTime<<endl;
cout<<"Number of Threads Created in Last Round:      "<<num_threads<<endl;
}
}