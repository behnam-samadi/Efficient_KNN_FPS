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
parallel_search_result round_result (64);

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
    for(int i = 0; i<num_query_points;i++){
        cout<<"KNN, Progress:" <<(float)i/num_query_points<<"\n";
        for (int j = 0; j<num_ref_points;j++)
        {
            distance[j] = calc_distance(query.data[i], reference.data[j], metric);
        }
        vector<int> topk = topK(distance, K);
        for(int c = 0; c<K;c++)
        {
            result[i][c] = topk[c];
        }
    }
return(result);
}

vector<int> KNN_one_row (Frame * reference, Frame * query, int K,dist_metric metric, int index){
    int num_ref_points = (*reference).data.size();
    int num_query_points = (*query).data.size();
    vector<int> result(K);
    vector<float>  distance (num_ref_points);
    int i = index;
        //cout<<"KNN, Progress:" <<(float)i/num_query_points<<"\n";
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
    Frame * reference;
    Frame * query;
    vector<int>* reference_order;
    vector<int> * query_order;
    int k;
    vector<vector<int>>* result;
     vector<float> *reference_projected;
     vector<float>* query_projected;
     int start_reference;
     int end_reference;
     int start_query;
     int num_ref_points;
     int end_query;
     vector<pthread_t*>* threads;
     vector<bool>* set_stat;
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





void exact_knn_projected(vector<vector<int>>* output,const Frame* reference,const vector<float> * reference_projected,const vector<int>* sorted_indices,vector<float>query, float query_projected, int nearest_index, int K, int row, int num_ref_points)
{
    if (nearest_index == 120755)
        {
            cout<<endl<<"injaaaaaaaa:::"<<endl<<"reference_projected "<<reference_projected<<endl;
            cout<<endl<<"injaaaaaaaa:::"<<endl<<"reference "<<reference<<endl;
            cout<<endl<<"injaaaaaaaa:::"<<endl<<"reference "<<reference<<endl;
            cout<<endl<<"injaaaaaaaa:::"<<endl<<"sorted_indices"<<sorted_indices<<endl;
            cout<<endl<<"injaaaaaaaa:::"<<endl<<"query_projected"<<query_projected<<endl;
            cout<<endl<<"injaaaaaaaa:::"<<endl<<"num_ref_points"<<num_ref_points<<endl;

        }
 //cout<<"//////////"<<reference<<endl;   
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

 //   if (nearest_index == 120755)
 //   {
 //       cout<<reference;
 //       cout<<endl<<"start_knn "<<start_knn<<"end_knn "<<end_knn<<endl;
 //       exit(0);
 //   }
    //cout<<endl<<start_knn<<" "<<end_knn;
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
    if (nearest_index == 120755)
        {
            cout<<max_dist<<"max_dist"<<endl;
        }
    if (right_arrow<num_ref_points)
        {
    while( abs( (*reference_projected)[right_arrow] - query_projected ) <= (sqrt(3)*max_dist)    )
    {
        dist = calc_distance((*reference).data[(*sorted_indices)[right_arrow]], query, Euclidean);
                   if (nearest_index == 120755)
        {
            cout<<"calculated dis: "<<dist<<"reference_projected "<<reference_projected<<endl;
        }
        calculated_distances_num++;
        if (dist < max_dist)
        {

            fout<<((*reference_projected)[right_arrow] - query_projected)<<" "<<max_dist <<endl;
            fout<<endl;
            //cout<<abs( (*reference_projected)[right_arrow] - query_projected )<<endl;
            //cout<<max_dist<<endl;
            knn.pop();
            knn.push(make_pair(dist, (*sorted_indices)[right_arrow]));
            max_dist = knn.top().first;
                   if (nearest_index == 120755)
        {
            cout<<"max_dist chnegd to "<<max_dist<<endl;
        }
        }
        right_arrow++;
        if (right_arrow == num_ref_points)
            break;
    }
}
if (left_arrow>0)
{
        while(abs((*reference_projected)[left_arrow] - query_projected) <= (sqrt(3)*max_dist))
    {
        dist = calc_distance((*reference).data[(*sorted_indices)[left_arrow]], query, Euclidean);
        calculated_distances_num++;
        if (dist < max_dist)
        {
            fout<< abs((*reference_projected)[left_arrow] - query_projected)<<" "<<max_dist <<endl;
            fout<<endl;
            knn.pop();
            knn.push(make_pair(dist, (*sorted_indices)[left_arrow]));
            max_dist = knn.top().first;
        }
        left_arrow--;
        if (left_arrow<0) break;
    }
}
int c = 0;
//if (nearest_index == 120755)
        
  //     {
    //        cout<<query_projected<<endl;
      //      print_vector_float(query);
        //    while(knn.size())
 //           {
  //          cout<<"row "<<c<<" -------------------"<<knn.top().second;
   //        (*output)[row][c++] = knn.top().second;
    //        knn.pop();
     //   }
      //  exit(0);
     //}
           if (nearest_index == 120755)
        {//cout<<endl<<"dist "<<dist<<endl;
    
    
        cout<<"response for row "<<row<<" :"<<endl;
        priority_queue<pair<float, int>> knn2 = knn;
        while(knn2.size())
        {
            cout<<endl<<"value: "<<knn2.top().second<<" "<<knn2.top().first<<endl;;
            knn2.pop();
        }
    
}
    while(knn.size())
    {
        
        (*output)[row][c++] = knn.top().second;
        knn.pop();
    }

    //cout<<endl<<"number of calculated_distances_num: "<<calculated_distances_num<<endl;    
}









    
void exact_knn_projected_(vector<vector<int>>* output,Frame* reference,vector<float> * reference_projected, vector<int>* sorted_indices,vector<float>query, float query_projected, int nearest_index, int K, int row, int num_ref_points)
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
    //cout<<endl<<start_knn<<" "<<end_knn;
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
    if (right_arrow<num_ref_points)
        {
    while( abs( (*reference_projected)[right_arrow] - query_projected ) <= (sqrt(3)*max_dist)    )
    {
        dist = calc_distance((*reference).data[(*sorted_indices)[right_arrow]], query, Euclidean);
        calculated_distances_num++;
        if (dist < max_dist)
        {
            fout<<((*reference_projected)[right_arrow] - query_projected)<<" "<<max_dist <<endl;
            fout<<endl;
            //cout<<abs( (*reference_projected)[right_arrow] - query_projected )<<endl;
            //cout<<max_dist<<endl;
            knn.pop();
            knn.push(make_pair(dist, (*sorted_indices)[right_arrow]));
            max_dist = knn.top().first;
        }
        right_arrow++;
        if (right_arrow == num_ref_points)
            break;
    }
}
if (left_arrow>0)
{
        while(abs((*reference_projected)[left_arrow] - query_projected) <= (sqrt(3)*max_dist))
    {
        dist = calc_distance((*reference).data[(*sorted_indices)[left_arrow]], query, Euclidean);
        calculated_distances_num++;
        if (dist < max_dist)
        {
            fout<< abs((*reference_projected)[left_arrow] - query_projected)<<" "<<max_dist <<endl;
            fout<<endl;
            knn.pop();
            knn.push(make_pair(dist, (*sorted_indices)[left_arrow]));
            max_dist = knn.top().first;
        }
        left_arrow--;
        if (left_arrow<0) break;
    }
}
int c = 0;
    while(knn.size())
    {
        (*output)[row][c++] = knn.top().second;
        //cout<<"change to result "<<row<<" "<<c<<" "<<knn.top().second<<endl;
        knn.pop();
    }
    //cout<<endl<<"number of calculated_distances_num: "<<calculated_distances_num<<endl;    
}





//exact_knn_projected(output, &reference_projected,&reference_order,20, 80,1564,  int num_ref_point)
//exact_knn_projected(&exact_fast_result,&reference,&reference_projected,   &reference_order, query.data[q],query_projected[q],nearest, k,num_ref_points);

spliting_state one_step_parallel_binary_search(vector<float> *reference_projected, vector<float>* query_projected,int start_reference, int end_reference, int start_query, int end_query, vector<vector<int>>* result, Frame* reference, vector<int>* reference_order, Frame* query, vector<int>* query_order, int k, int num_ref_points, vector<bool> *set_stat)
{
    
    int middle_index = (start_reference + end_reference)/2;
    float middle_value = (*reference_projected)[middle_index];
    spliting_result split = binary_search_split(query_projected, start_query, end_query, middle_value);
    int divider1 = split.divider1;
    int divider2 = split.divider2;
    round_result.set_value(middle_index, divider1, divider2);
    for(int d = divider1; d< divider2;d++)
    {        
     exact_knn_projected     (result                     ,reference       ,reference_projected                ,reference_order              ,query->data[(*query_order)[d]],(*query_projected)[d],middle_index, k,d,num_ref_points);    
     (*set_stat)[d] = true;

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
    vector<float> * reference_projected = data->reference_projected;
    Frame * reference = data->reference;
    //cout<<"------------------------------"<<reference;
    Frame * query = data->query;
    vector<float> * query_projected = data->query_projected;
    vector<int>* reference_order = data->reference_order;
    vector<int> * query_order = data->query_order;
    //vector<int> ;
    int k = data->k;
    int start_reference = data->start_reference;
    int end_reference = data->end_reference;
    int start_query = data->start_query;
    vector<bool>* set_stat = data->set_stat;
    int end_query = data->end_query;
    int num_ref_points = data->num_ref_points;
    //cout<<"----------line708--------"<<num_ref_points;
    pthread_mutex_t* push_mutex = data->push_mutex;
    vector<vector<int>>* result = data->result;
    vector<pthread_t*>* threads = data->threads;
    int middle_index;
    spliting_state state;
    do{   
        //cout<<start_reference<<" "<<end_reference<<" "<<start_query<<" "<<end_query<<endl;
    if ((end_reference - start_reference)<2)
    {
        for (int q = start_query ; q<= end_query;q++)
        {
            int nearest = binary_search(reference_projected, (*query_projected)[q], start_reference, end_reference);
            round_result.set_value(nearest, q, q+1);
         exact_knn_projected     (result                     ,reference       ,reference_projected                ,reference_order              ,query->data[(*query_order)[q]],(*query_projected)[q],nearest, k,q,num_ref_points);       
         (*set_stat)[q] = true;
        }
        delete data;
        return NULL;
    }
    state = one_step_parallel_binary_search(reference_projected , query_projected , start_reference, end_reference , start_query, end_query, result, reference, reference_order, query, query_order, k, num_ref_points, set_stat);
    middle_index = state.middle_index;
        if(state.right_size > 0)
        {

            pthread_t* temp = new pthread_t;           
            pthread_mutex_lock(push_mutex);
            threads->push_back(temp);
            pthread_mutex_unlock(push_mutex);
            thread_data* args = new thread_data;
            args->reference = reference;
            args->reference_projected = reference_projected;
            args->query_projected = query_projected;
            args->query = query;
            args->query_order = query_order;
            args->reference_order = reference_order;
            args->k = k;
            args->set_stat = set_stat;
            args->start_reference = middle_index;
            args->end_reference = end_reference;
            args->start_query = state.divider2;
            args->num_ref_points = num_ref_points;
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
            int nearest = binary_search(reference_projected, (*query_projected)[start_query], start_reference, end_reference);
            exact_knn_projected     (result                     ,reference         ,reference_projected                ,reference_order              ,query->data[(*query_order)[start_query]],(*query_projected)[start_query],nearest, k,start_query,num_ref_points);    
            round_result.set_value(nearest, start_query, start_query+1);
            //cout<<"seg fault at (maybe) "<<start_query;
            (*set_stat)[start_query] = true;
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
    int num_query_points_orig = num_query_points;
    num_query_points = 64;
    int round_size = num_query_points;
    
    vector<float> reference_projected(num_ref_points);
    vector<float> query_projected(round_size);
    for (int i =0 ; i<num_ref_points;i++)
    {
        reference_projected[i] = 0;
        for (int j = 0; j<reference.data[0].size();j++)
        {
        reference_projected[i] += reference.data[i][j];
        }
    }
    for (int i =0 ; i<round_size;i++)
    {
        query_projected[i] = 0;
        for (int j = 0; j<query.data[0].size();j++)
        {
        query_projected[i] += query.data[i][j];
        }
    }
    

    vector<int> reference_order(num_ref_points);
    vector<int> reference_order_reverse (num_ref_points);
    iota(reference_order.begin(),reference_order.end(),0); //Initializing
    sort( reference_order.begin(),reference_order.end(), [&](int i,int j){return reference_projected[i]<reference_projected[j];} );
    sort( reference_projected.begin(),reference_projected.end());


    vector<int> query_order(round_size);
    iota(query_order.begin(),query_order.end(),0); //Initializing
    sort( query_order.begin(),query_order.end(), [&](int i,int j){return query_projected[i]<query_projected[j];} );
    sort( query_projected.begin(),query_projected.end());
    cout<<"bordarha sakhte shodand"<<endl;

    //print_vector(query_order);
  int q = 0;  
int k = 5;


int test_q = 25;
int nearest_for_zero = binary_search(&reference_projected, query_projected[test_q], 0 , num_ref_points);
cout<<endl<<"nearest_for_zero"<<nearest_for_zero<<endl;

vector<vector<int>> temp  (1 , vector<int> (k, 0));

    exact_knn_projected(&temp ,&reference,&reference_projected,&reference_order, query.data[query_order[test_q]]              ,query_projected[test_q],nearest_for_zero, k,q,num_ref_points);    
    vector<int> KNN_one_row_test_single =  KNN_one_row(&reference,&query,k,Euclidean,query_order[test_q]);
    print_vector(KNN_one_row_test_single);
    //exact_knn_projected(result,reference ,reference_projected ,reference_order , query->data[(*query_order)[start_query]],(*query_projected)[start_query],nearest, k,start_query,num_ref_points);    
    print_vector_float(query.data[query_order[test_q]]);
    cout<<endl<<query_projected[test_q];
    

    
    

    for (int i = 0 ; i < reference_order.size(); i++)
    {
        reference_order_reverse[reference_order[i]] = i;
    }
 
    vector<vector<int>> exact_fast_result  (num_query_points , vector<int> (k, 0));
    //vector<vector<int>> knn_result;
    vector<int> row_result;
    //for (int q =0 ; q<num_query_points;q++)
    //{
        //row_result = KNN_one_row (Frame reference, Frame query, int K,dist_metric metric, int index){
        //row_result = KNN_one_row(&reference , &query, k , Euclidean, query_order[q]);
        //cout<<endl<<q<<"'th result:"<<endl;
        //print_vector(row_result);
    //}
    //knn_result = KNN(reference, query, k, Modified_Manhattan,num_query_points);
//(Frame reference, Frame query, int K,string metric,  int num_query=0)
    //cout<<endl<<endl<<"javabe bargashte shode:"<<endl;
    //print_vector_2D(knn_result);
    //cout<<"in bood"<<endl;
    //exit(0);

    
    vector<bool> set_stat(num_query_points);
    for (int q= 0 ; q<num_query_points; q++)
    {
        set_stat[q] = false;
    }
    double runTime = -omp_get_wtime();

    thread_data* args = new thread_data;
    
    vector<pthread_t*> round_threads;
    args->reference = &reference;
    args->query = &query;
    args->reference_projected = &reference_projected;
    args->k = k;
    args->query_order = &query_order;
    args->set_stat = &set_stat;
    //args->reference = &test_reference;
    args->query_projected = &query_projected;
    //args->query = &test_query;
    args->start_reference = 0;
    args->end_reference = reference.data.size()-1;
    args->num_ref_points = num_ref_points;
    //args->end_reference = 39;
    args->start_query = 0;
    args->end_query = round_size-1;
    args->result = &exact_fast_result;
    args->threads = &round_threads;
    args->reference_order = &reference_order;
    args->push_mutex = new pthread_mutex_t;

    pthread_t * main_thread = new pthread_t;
    round_threads.push_back(main_thread);
    pthread_mutex_init(args->push_mutex,0);
    cout<<endl<<args->reference<<endl;
    pthread_create( main_thread, NULL, parallel_binary_search, (void*)args);
    cout<<endl<<endl<<endl<<"start running parallel code"<<endl;
    //exact_knn_projected(result,reference,reference_projected,&reference_order,query.data[query_order[q]],query_projected[q],nearest, k,q,num_ref_points);    
    //args.threads->push_back(new pthread_t);
    //spliting_state one_step_parallel_binary_search(vector<float> *reference, vector<float>* query,int start_reference, int end_reference, int start_query, int end_query, parallel_search_result* result)
    //spliting_state temp_state =  one_step_parallel_binary_search(&reference_projected, &query_projected, 60507 , num_ref_points, 0,round_size, &round_result);
    //print_vector_float(query_projected);
    //cout<<"middle index: "<<60507<<endl;
    //cout<<"middle point "<<reference_projected[60507]<<endl;
    //return(0);

    //cout<< endl<<temp_state.left_size<<" "<<temp_state.middle_size<<" "<< temp_state.right_size<<" "<<temp_state.divider1<<" "<<temp_state.divider2<<endl;
    //return (0);
    
    
    //parallel_binary_search((void*)(args));

    //while(!(round_result.all_set()));
        //{cout<<endl<<"waiting";}
    bool round_working = true;
    while(!(round_result.all_set()));
    while(round_working)
    {
        bool finish = true;
        for (int result_part = 0 ; result_part < num_query_points; result_part++)
        {
            if (set_stat[result_part] == false)
            {
                finish = false;
                break;
            }
        }
        round_working = !finish;
    }
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


    

//print_vector_2D(exact_fast_result);
//exit(0);
cout<<"whole comparision"<<endl;
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
    //cout<<"proposed result:"<<endl;
    //print_vector(exact_fast_result[q]);
    //cout<<"correct answer for "<<q<<"'th element"<<endl;
    //print_vector(KNN_one_row_test);
    //cout<<"matches"<<matches<<endl;
    if (matches == k)
        score++;
}
cout<<"score" <<score;

    

cout<<"az inja be bad"<<endl;
for (int q = 0 ; q<round_size;q++)
    {
        //cout<<endl<<q<<"'th element: "<<round_result.values[q];
        //cout<<endl<<result;
    }

cout<<"temp in ast:"<<endl;
 //   print_vector_2D(temp);
    exit(0);
    cout<<endl<<query_projected[q];
    print_vector_float(query.data[query_order[q]]);
    cout<<"aya injast?"<<endl;
    vector<int> KNN_one_row_test =  KNN_one_row(&reference,&query,k,Euclidean,query_order[0]);

    print_vector(KNN_one_row_test);
    exit(0);    


for (int q = 0 ; q<round_size;q++)
    {
        //cout<<endl<<q<<"'th element: "<<reference_order[round_result.values[q]];
        //cout<<endl<<result;
    }
    
    exit(0);
    
    //delete args->push_mutex;
    //delete main_thread;    
    //float elapsed = clock() - begin_time;

    runTime += omp_get_wtime();
    cout<<endl<<runTime<<endl;


exit(0);
/*
for (int q = 0; q < num_query_points; q++)
{
    int score = 0;
    int nearest = binary_search(&reference_projected, query_projected[q], 0, num_ref_points-1);
    exact_knn_projected(&exact_fast_result,&reference,&reference_projected,&reference_order,query.data[query_order[q]],query_projected[q],nearest, k,q,num_ref_points);    
    knn_result = KNN(reference, query, k, Euclidean,query_order[q]);
    //exit();
    //iota(exact_fast_result[q].begin(),exact_fast_result[q].end(),0); //Initializing
    sort(exact_fast_result[q].begin(),exact_fast_result[q].end());
    //iota(knn_result.begin(),knn_result.end(),0); //Initializing
    //sort(knn_result.begin(),knn_result.end(), [&](int i,int j){return knn_result[i]<knn_result[j];} );
    sort(knn_result.begin(),knn_result.end());
    for (int c = 0; c<k; c++)
    {
        if (exact_fast_result[q][c] == knn_result[c])
        {
            score++;
        }
    }
    if (score < k) 
    {
        cout<<endl<<q<<"'th has wrong answer"<<score<< endl;
        //exit(0);
    }
    cout<<"about "<<q<<"'th query: "<<endl;
    cout<<"proposed: "<<endl;
    print_vector(exact_fast_result[q]);
    cout<<"ground truth: "<<endl;
    //vector<int> KNN (Frame reference, Frame query, int K,dist_metric metric, int index){
    
    print_vector(knn_result);
    cout<<"end "<<q<<"'th query: "<<endl;
    
}
print_vector_2D(exact_fast_result);
exit(0);
//knn_result = KNN(reference, query, k, Modified_Manhattan,num_query_points);
//test area
vector<vector<int>>* output;
int nearest = 118;
//void exact_knn_projected(vector<vector<float>>* output, vector<float> * reference_projected,vector<int>* reference_order, int nearest_index, int K, int row, int num_ref_points)
//exact_knn_projected(output,&reference,query.data[nearest], &reference_projected,&reference_order,query_projected[nearest], nearest, 10,80,num_ref_points);
exit(0);
//test area
    
    
    vector<float> test_reference  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40};
    vector<float> test_query = {2.2, 3.5, 21.6, 35.7, 90};
    spliting_result split = binary_search_split(&test_query, 0, 4, 93);
    cout<<split.divider1<<" "<<split.divider2<<" ";
    
    //float begin_time = clock();
    exit(0);   
cout<<"obtained results"<<endl;
    for (int q = 0 ; q<round_size;q++)
    {
        cout<<endl<<q<<"'th element: "<<reference_order[round_result.values[q]];
        //cout<<endl<<result;
    }
    cout<<endl;
cout<<"correct results:"<<endl;
    for (int q = 0 ; q<round_size;q++)
    {
        
        int result = binary_search (&reference_projected, query_projected[q], 0, num_ref_points);
        cout<<endl<< q<<"'th element "<<reference_order[result];
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
/*    cout<<endl<<"54's result:"<<endl;
    //12640
    //print_vector(knn_result[54]);
    //return(0);
    cout<<"ine:"<<query_order[0];
    cout<<"va in"<<reference_order[59];
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
    //cout<<knn_result[0].size();
    //print_vector_2D(knn_result);
    //vector<vector<int>>  ground_truth = KNN(reference, query, k, Euclidean,num_query_points);
    /*int mathces = 0;
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
*/
}