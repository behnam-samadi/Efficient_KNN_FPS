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
using namespace std;
vector<int> sorted_indices_reverse (211000);
//cleaning
ofstream fout("in_and_out.txt");

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

class parallel_search_result{  
    public:
    vector <int>  values;
    vector <bool> inits;
    parallel_search_result(int size)
    {
        values = vector<int> (size);
        inits = vector<bool> (size);
        for (int i = 0 ;i<size;i++)
        {
            inits[i] = 0;
        }
    }
    void set_value(int value, int start_index, int end_index)
    {
        for (int i = start_index; i<end_index; i++)
        {
            if (this->inits[i] == 1)
            {
                cout<<endl<<endl<<endl<<"---Double Assigning!!!!!--- for element:"<<i<<" which has value "<<values[i]<<"and is becoming "<<value;
            }
            this->values[i] = value;
            this->inits[i] = 1;
        }
    }
    bool all_set()
    {
        bool result = 1;
        for (int i = 0 ; i<this->inits.size();i++)
            if (this->inits[i] == 0) result = 0;
        return result;
    }
};


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


void print_vector (vector<int> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<v[i]<<" ";
    }
    cout<<endl;
}


vector<vector<int>> KNN (Frame reference, Frame query, int K,dist_metric metric,  int num_query=0){
    int num_ref_points = reference.data.size();
    int num_query_points = query.data.size();
    if (!(num_query == 0)) num_query_points = num_query;
    vector<vector<int>> result  (num_query_points , vector<int> (K, 0));
    vector<float>  distance (num_ref_points);
    for(int i = 2708; i<2709;i++){
        cout<<"KNN, Progress:" <<(float)i/num_query_points<<"\n";
        for (int j = 0; j<num_ref_points;j++)
        {
            distance[j] = calc_distance(query.data[i], reference.data[j], metric);
        }
        vector<int> topk = topK(distance, K);
        cout<<"in javabe nahayi ast:"<<endl;
        print_vector(topk);
        
        for(int c = 0; c<K;c++)
        {
            cout<<"dis"<<endl;
            //result[i][c] = topk[c];
            cout<<endl<<c<<"'th answer: "<<topk[c];
        }

        
    }
return(result);
}




void print_vector_bool (vector<bool> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<v[i]<<" ";
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
struct thread_data
{
 vector<float> *reference;
 vector<float>* query;
 int start_reference;
 int end_reference;
 int start_query;
 int end_query;
 parallel_search_result* result;
 vector<pthread_t*>* threads;
 pthread_mutex_t *push_mutex;
};

int binary_search (vector<float>* reference, float query, int begin, int end)
{
    int length = end - begin+1;
    int end_orig = end;
    int middle_index = (begin + end) / 2;
    float middle = (*reference)[(int)((begin + end) / 2)];
    while (end >= begin)
    {
        middle_index = (begin + end) / 2;
        middle = (*reference)[(int)((begin + end) / 2)];

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
            diff2 = abs(query - (*reference)[(middle_index+1)]);
        }
        else {
            diff2 =numeric_limits<float>::max() ;
        }
        if (middle_index > 0)
        {
            diff3 = abs(query - (*reference)[middle_index-1]);
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

struct query_point
{
    vector<float> point;
    int index;
};

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


//exact_knn_projected(output, &sum_cordinates,&sorted_indices,20, 80,1564,  int num_ref_point)
//exact_knn_projected(&result_projected,&reference,&sum_cordinates,   &sorted_indices, query.data[q],sum_cordinates_query[q],nearest, k,num_ref_points);
void exact_knn_projected(vector<vector<int>>* output,Frame* reference,vector<float> * reference_projected, vector<int>* sorted_indices, vector<float>query, float query_projected, int nearest_index, int K, int row, int num_ref_points)
{
    int start_knn = nearest_index;
    int end_knn = nearest_index;
    //float middle_value = (*reference_projected)[nearest_index];
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
        if ((abs((*reference_projected)[start_knn-1]-query_projected)) < (abs((*reference_projected)[end_knn+1]-query_projected)))
        {
            start_knn--;
        }
        else
        {
            end_knn++;
        }
    }
    cout<<endl<<start_knn<<" "<<end_knn;
    //exit(0);
    //int max_index = start_knn;
    float max_dist = calc_distance((*reference).data[(*sorted_indices)[start_knn]], query, Euclidean);
    float dist;
    int calculated_distances_num = 0;
    priority_queue<pair<float, int>> knn;
    for(int c = start_knn; c<= end_knn; c++)
    {
        dist = calc_distance((*reference).data[(*sorted_indices)[c]], query, Euclidean);
        calculated_distances_num ++;
        knn.push(make_pair(dist, (*sorted_indices)[c]));
        if (dist > max_dist)
        {
            max_dist = dist;
            //max_index = c;
        }
    }

    int right_arrow = end_knn+1;
    int left_arrow = start_knn-1;

    max_dist = knn.top().first;
    cout<<"first..."<<endl;

      //while(knn.size())
    //{
        //cout<<endl<<knn.top().first<<" "<<knn.top().second;
        //knn.pop();
    //}
    cout<<"and then..."<<endl;
//                   cout<<endl<<"chape priority_queue:";
//                while(knn.size())
//    {
//        cout<<endl<<knn.top().first<<" "<<knn.top().second<<" "<<sorted_indices_reverse[knn.top().second];
//        knn.pop();
//    }
//    exit(0);
        //right_arrow = 141;
        //cout<<"do joor:"<<endl;
        //dist = calc_distance((*reference).data[(*sorted_indices)[right_arrow]], query, Euclidean);
        //vector<float> test1 = (*reference).data[(*sorted_indices)[right_arrow]];
        //print_vector_float(test1);
        //print_vector_float(query);
        //exit(0);
        //cout<<"in 3 ta"<<endl;
        //cout<<endl<<abs((((*reference).data[(*sorted_indices)[right_arrow]][0] + (*reference).data[(*sorted_indices)[right_arrow]][1] + (*reference).data[(*sorted_indices)[right_arrow]][2])) - (query[0] + query[1] + query[2]));
        //cout<<endl<<abs((*reference_projected)[right_arrow] - query_projected);
        //dist = calc_distance((*reference).data[(*sorted_indices)[right_arrow]], query, Euclidean);
        //cout<<endl<<dist;
        
    while( abs( (*reference_projected)[right_arrow] - query_projected ) <= (sqrt(2)*max_dist)    )
    {
        dist = calc_distance((*reference).data[(*sorted_indices)[right_arrow]], query, Euclidean);
        calculated_distances_num++;
        if (dist < max_dist)
        {
            fout<<endl<<"R: ("<<knn.top().second<<","<<knn.top().first<<","<<(sorted_indices_reverse)[knn.top().second]<<") is going out and ("<<(*sorted_indices)[right_arrow]<<","<<dist<<","<<right_arrow<<") is going in because"<<dist<<"is smaller than "<<max_dist<<endl;
            fout<<((*reference_projected)[right_arrow] - query_projected)<<" "<<max_dist <<endl;
            fout<<endl;
           
            cout<<abs( (*reference_projected)[right_arrow] - query_projected )<<endl;
            cout<<max_dist<<endl;

          
            knn.pop();
            max_dist = knn.top().first;
            knn.push(make_pair(dist, (*sorted_indices)[right_arrow]));
            //cout<<endl<<"in top while";
            //cout<<endl<<dist<<" "<<right_arrow;
 //                   cout<<endl<<"chape priority_queue:";
 //               while(knn.size())
 //   {
 //       cout<<endl<<knn.top().first<<" "<<knn.top().second<<" "<<sorted_indices_reverse[knn.top().second];
 //       knn.pop();
 //   }
 //   exit(0);


        }
        right_arrow++;
        if (right_arrow == num_ref_points-1)
            break;
    }
        while(abs((*reference_projected)[left_arrow] - query_projected) <= (sqrt(2)*max_dist))
    {
        dist = calc_distance((*reference).data[(*sorted_indices)[left_arrow]], query, Euclidean);
        calculated_distances_num++;
        if (dist < max_dist)
        {
            
        fout<<endl<<"L: ("<<knn.top().second<<","<<knn.top().first<<","<<(sorted_indices_reverse)[knn.top().second]<<") is going out and ("<<(*sorted_indices)[left_arrow]<<","<<dist<<","<<left_arrow<<") is going in because"<<dist<<"is smaller than "<<max_dist<<endl;
            fout<< abs((*reference_projected)[left_arrow] - query_projected)<<" "<<max_dist <<endl;
            fout<<endl;
            knn.pop();
            max_dist = knn.top().first;
            knn.push(make_pair(dist, (*sorted_indices)[left_arrow]));
            //cout<<endl<<"in down while";
            //cout<<endl<<dist<<" "<<left_arrow;
        }
        left_arrow--;
        if (left_arrow==0) break;
    }

//    while(abs((*reference_projected)[left_arrow] - (*reference_projected)[nearest_index] <= max_dist)
//    {

  //  }
    cout<<endl;
      while(knn.size())
    {
        cout<<endl<<knn.top().first<<" "<<sorted_indices_reverse[knn.top().second];
        knn.pop();
    }

    
    exit(0);
cout<<endl<<"result:"<<endl;
    while(knn.size())
    {
        cout<<endl<<knn.top().first<<" "<<knn.top().second;
        knn.pop();
    }
    cout<<endl<<"number of calculated_distances_num: "<<calculated_distances_num<<endl;

    
}

spliting_state one_step_parallel_binary_search(vector<float> *reference, vector<float>* query,int start_reference, int end_reference, int start_query, int end_query, parallel_search_result* result)
{
        
    int middle_index = (start_reference + end_reference)/2;
    float middle_value = (*reference)[middle_index];
    spliting_result split = binary_search_split(query, start_query, end_query, middle_value);
    int divider1 = split.divider1;
    int divider2 = split.divider2;
    result->set_value(middle_index, divider1, divider2);
    
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
    vector<float> * reference = data->reference;
    vector<float> * query = data->query;
    int start_reference = data->start_reference;
    int end_reference = data->end_reference;
    int start_query = data->start_query;
    int end_query = data->end_query;
    pthread_mutex_t* push_mutex = data->push_mutex;
    parallel_search_result* result = data->result;
    vector<pthread_t*>* threads = data->threads;
    int middle_index;
    spliting_state state;
    int task_size;
    do{   
        //cout<<start_reference<<" "<<end_reference<<" "<<start_query<<" "<<end_query<<endl;
    if ((end_reference - start_reference)<2)
    {
        for (int q = start_query ; q<= end_query;q++)
        {
            int single_result = binary_search(reference, (*query)[q], start_reference, end_reference);
            result->set_value(single_result, q, q+1);
            
        }
        delete data;
        
        return NULL;
    }
    state = one_step_parallel_binary_search(reference , query , start_reference, end_reference , start_query, end_query, result);
    middle_index = state.middle_index;
        if(state.right_size > 0)
        {
            pthread_t* temp = new pthread_t;           
            pthread_mutex_lock(push_mutex);
            threads->push_back(temp);
            pthread_mutex_unlock(push_mutex);
            thread_data* args = new thread_data;
            args->reference = reference;
            args->query = query;
            args->start_reference = middle_index;
            args->end_reference = end_reference;
            args->start_query = state.divider2;
            args->end_query = end_query;
            args->result = result;
            args->threads = threads;
            args->push_mutex = push_mutex;
            if (args->end_query >= args->start_query)
            {
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
            int single_result = binary_search(reference, (*query)[start_query], start_reference, end_reference);
            result->set_value(single_result, start_query, start_query+1);
            
        }
        delete data;
        
}
int main()
{
      
    
    int frame_channels = 3;
    Frame reference = read_data("0000000000.bin", 4, frame_channels);
    Frame query = read_data("0000000001.bin", 4, frame_channels);
    int num_ref_points = reference.data.size();
    int num_query_points = query.data.size();
    num_query_points = num_query_points;
    int round_size = num_query_points;
    
    vector<float> sum_cordinates(num_ref_points);
    vector<float> sum_cordinates_query(round_size);
    for (int i =0 ; i<num_ref_points;i++)
    {
        sum_cordinates[i] = 0;
        for (int j = 0; j<reference.data[0].size();j++)
        {
        sum_cordinates[i] += reference.data[i][j];
        }
    }
    for (int i =0 ; i<round_size;i++)
    {
        sum_cordinates_query[i] = 0;
        for (int j = 0; j<query.data[0].size();j++)
        {
        sum_cordinates_query[i] += query.data[i][j];
        }
    }
    

    vector<int> sorted_indices(num_ref_points);
    iota(sorted_indices.begin(),sorted_indices.end(),0); //Initializing
    sort( sorted_indices.begin(),sorted_indices.end(), [&](int i,int j){return sum_cordinates[i]<sum_cordinates[j];} );
    sort( sum_cordinates.begin(),sum_cordinates.end());

cout<<endl<<endl<<sum_cordinates[120555];
    //exit(0);

    vector<int> sorted_query(round_size);
    iota(sorted_query.begin(),sorted_query.end(),0); //Initializing
    sort( sorted_query.begin(),sorted_query.end(), [&](int i,int j){return sum_cordinates_query[i]<sum_cordinates_query[j];} );
    sort( sum_cordinates_query.begin(),sum_cordinates_query.end());


num_query_points = 125;
int k = 6;
vector<vector<int>> result_projected  (num_query_points , vector<int> (k, 0));

cout<<"injast:"<<endl;
for (int c= 100; c<=105;c++)
{
    cout<<endl<<sorted_indices[c];
}
cout<<endl<<sorted_query[120];


for (int i = 0 ; i < sorted_indices.size(); i++)
{
    sorted_indices_reverse[sorted_indices[i]] = i;
}
cout<<endl<<calc_distance(reference.data[6335], query.data[2708], Euclidean);
//exit(0);
cout<<"inha hatand:"<<endl;
cout<<endl<<sorted_indices_reverse[4525];
cout<<endl<<sorted_indices_reverse[881];
cout<<endl<<sorted_indices_reverse[6334];
cout<<endl<<sorted_indices_reverse[2713];
cout<<endl<<sorted_indices_reverse[884];
cout<<endl<<sorted_indices_reverse[6335];

//cout<<endl<<sorted_indices_reverse[2756];
//exit(0);
vector<vector<int>> knn_result;
knn_result = KNN(reference, query, k, Euclidean,num_query_points);
//exit(0);

for (int q = 120; q < num_query_points; q++)
{
    cout<<endl<< sum_cordinates_query[q]<<" sum_cordinates_query[q]";
    //exit(0);
    int nearest = binary_search(&sum_cordinates, sum_cordinates_query[q], 0, num_ref_points-1);
    cout<<endl<<nearest;
    //exit(0);
    exact_knn_projected(&result_projected,&reference,&sum_cordinates,&sorted_indices, query.data[sorted_query[q]],sum_cordinates_query[q],nearest, k,q,num_ref_points);    
    exit(0);
    cout<<"and correct result:"<<endl;
    knn_result = KNN(reference, query, k, Modified_Manhattan,num_query_points);
    print_vector(knn_result[25]);
    exit(0);
}
//test area
vector<vector<int>>* output;
int nearest = 118;
//void exact_knn_projected(vector<vector<float>>* output, vector<float> * reference_projected,vector<int>* sorted_indices, int nearest_index, int K, int row, int num_ref_points)
//exact_knn_projected(output,&reference,query.data[nearest], &sum_cordinates,&sorted_indices,sum_cordinates_query[nearest], nearest, 10,80,num_ref_points);
exit(0);


//test area
    
    
    vector<float> test_reference  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40};
    vector<float> test_query = {2.2, 3.5, 21.6, 35.7, 90};
    spliting_result split = binary_search_split(&test_query, 0, 4, 93);
    cout<<split.divider1<<" "<<split.divider2<<" ";


    double runTime = -omp_get_wtime();

    //float begin_time = clock();

    thread_data* args = new thread_data;
    parallel_search_result round_result (round_size);
    vector<pthread_t*> round_threads;
    cout<<"round_threads"<<&round_threads<<endl;
    args->reference = &sum_cordinates;
    //args->reference = &test_reference;
    args->query = &sum_cordinates_query;
    //args->query = &test_query;
    //args->start_reference = 0;
    args->end_reference = reference.data.size()-1;
    //args->end_reference = 39;
    args->start_query = 0;
    args->end_query = round_size-1;
    args->result = &round_result;
    args->threads = &round_threads;
    args->push_mutex = new pthread_mutex_t;
    pthread_t * main_thread = new pthread_t;
    round_threads.push_back(main_thread);
    pthread_mutex_init(args->push_mutex,0);
    pthread_create( main_thread, NULL, parallel_binary_search, (void*)args);
    cout<<endl<<endl<<endl<<"start running parallel code"<<endl;
    //args.threads->push_back(new pthread_t);
    //spliting_state one_step_parallel_binary_search(vector<float> *reference, vector<float>* query,int start_reference, int end_reference, int start_query, int end_query, parallel_search_result* result)
    //spliting_state temp_state =  one_step_parallel_binary_search(&sum_cordinates, &sum_cordinates_query, 60507 , num_ref_points, 0,round_size, &round_result);
    //print_vector_float(sum_cordinates_query);
    //cout<<"middle index: "<<60507<<endl;
    //cout<<"middle point "<<sum_cordinates[60507]<<endl;
    //return(0);

    //cout<< endl<<temp_state.left_size<<" "<<temp_state.middle_size<<" "<< temp_state.right_size<<" "<<temp_state.divider1<<" "<<temp_state.divider2<<endl;
    //return (0);
    
    
    //parallel_binary_search((void*)(args));

    while(!(round_result.all_set()));
        //{cout<<endl<<"waiting";}
    for(int t = 0; t<round_threads.size();t++)
    {
        pthread_join(*(round_threads[t]),NULL);
    }
    //delete args;
    for (int tn = 0; tn < round_threads.size();tn++)
    {

        delete round_threads[tn];
        //cout<<endl<<"for thread "<<tn<<endl;
    }
    //delete args->push_mutex;
    //delete main_thread;    
    //float elapsed = clock() - begin_time;
    runTime += omp_get_wtime();
    cout<<endl<<runTime<<endl;
    exit(0);   

cout<<"obtained results"<<endl;
    for (int q = 0 ; q<round_size;q++)
    {
        cout<<endl<<q<<"'th element: "<<sorted_indices[round_result.values[q]];
        //cout<<endl<<result;
    }
    cout<<endl;



cout<<"correct results:"<<endl;
    for (int q = 0 ; q<round_size;q++)
    {
        
        int result = binary_search (&sum_cordinates, sum_cordinates_query[q], 0, num_ref_points);
        cout<<endl<< q<<"'th element "<<sorted_indices[result];
    }
    cout<<endl;

    exit(0);

    //fout<<endl<<endl<<endl<<"final result"<<endl;
    //cout<<endl<<round_threads.size()<<endl;
    
    //print_vector(round_result.values);
    for (int p = 0 ;p <round_result.values.size();p++)
    {
        if (!(round_result.inits[p])) {cout<<endl<<"falut!!!"<<endl;exit(0);}
        cout<<endl<<p<<": "<<round_result.values[p]<<endl;
    }
    

return(0);
/*
    thread_data* data = (thread_data*)(data_void);
    vector<float> * reference = data->reference;
    vector<float> * query = data->query;
    int start_reference = data->start_reference;
    int end_reference = data->end_reference;
    int start_query = data->start_query;
    int end_query = data->end_query;
    parallel_search_result* result = data->result;
    vector<pthread_t*>* threads = data->threads;
*/






    //vector<vector<int>> knn_result = KNN(reference, query, k, Modified_Manhattan,num_query_points);

    //cout<<knn_result.size()<<endl;
    //cout<<knn_result[0].size();
    cout<<endl<<"54's result:"<<endl;
    //12640
    print_vector(knn_result[54]);
    //return(0);


    cout<<"ine:"<<sorted_query[0];
    cout<<"va in"<<sorted_indices[59];


    return(0);
    //int k = 2;
    //vector<vector<int>> knn_result = KNN(reference, query, k, "Modified_Manhattan",num_query_points);
    //cout<<knn_result.size()<<endl;
    //cout<<knn_result[0].size();
    //print_vector_2D(knn_result);
    //cout<<"result"<<endl;
    int num_round = num_query_points / round_size;
    for (int round = 0 ; round < num_round;round++)
    {
        
    }

    exit(0);
    //int k = 1;
    //vector<vector<int>> knn_result = KNN(reference, query, k, "Modified_Manhattan",num_query_points);
    cout<<knn_result.size()<<endl;
    cout<<knn_result[0].size();
    print_vector_2D(knn_result);
    vector<vector<int>>  ground_truth = KNN(reference, query, k, Euclidean,num_query_points);
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

}