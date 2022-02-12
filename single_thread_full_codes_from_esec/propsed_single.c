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

enum dist_metric
{
    Modified_Manhattan,
    Euclidean,
    Manhattan
};

class Frame{
    //later: change to private
public:
    vector<vector<double>> row_data;
    int num_points; 
    double ** data;
    Frame(string file_adress, int max_points=0)
    {
    ifstream fin(file_adress, ios::binary);
    fin.seekg(0, ios::end);
    size_t num_elements = fin.tellg() / sizeof(double);
    cout<<file_adress<<file_adress<< num_elements<<endl;
    if (max_points!=0) num_elements = (max_points*point_dim);
    num_points = num_elements/point_dim;
    cout<<num_points<<endl;
    fin.seekg(0, ios::beg);
    //fin.read(reinterpret_cast<char*>(&data_temp[0]), num_elements*sizeof(double))
    row_data = vector<vector<double>> (num_points , vector<double> (point_dim, 0));
    for (int c = 0 ; c<num_points; c++)
    {
        if (c%200 == 0) 
            {cout<<c<<endl;}
        fin.read(reinterpret_cast<char*>(&row_data[c][0]), point_dim*sizeof(double));
        //cout<<data[c][0]<<endl;
    }
    //cout<<"first part done";
    //exit(0);
    allocate_data();
    }
    void allocate_data()
    {
        //allocating 
        //cout<<endl<<num_points*(point_dim+1)<<endl;
        double * temp = new double [num_points*(point_dim+1)];
        //cout<<"new double created for "<<num_points*(point_dim+1)<<endl;
        data = new double*[num_points];
        for (int i = 0 ; i < num_points;i++)
        {
            data[i] = (temp+i*(point_dim+1));
        }
    }
};


void print_double_2d (double ** v, int X, int Y){
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


void print_vector_double (vector<double> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<v[i]<<" ";
    }
    cout<<endl;
}


double calc_distance (double *v1, vector<double>v2, dist_metric type)
{
    //cout<<"distance of ";
    //print_vector_double(v1);
    //print_vector_double(v2);
    if (type == Modified_Manhattan)
    {
        double sum1 = 0;
        double sum2 = 0;
        for(int i = 0; i<point_dim;i++)
            sum1+=v1[i];
        for(int i = 0; i<point_dim;i++)
            sum2+=v2[i];
        return (abs(sum2 - sum1));
    }
    else
    {
        double sum = 0;
        for(int i = 0; i<point_dim;i++)
        {
            if (type==Euclidean)
            sum+= pow(abs(v1[i] - v2[i]), 2);
            if (type==Manhattan)
            sum+= abs(v1[i] - v2[i]);
        }
        double result = sum;
        if (type == Euclidean)
            result = sqrt(result);
        //cout<<result;
        //exit(0);
        return(result);
        }
}


vector<int> topK(vector<double> input, int K){
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
    int num_ref_points = reference->num_points;
    int num_query_points = query->num_points;
    vector<int> result(K);
    vector<double>  distance (num_ref_points);
    int i = index;
        //cout<<"KNN, Progress:" <<(double)i/num_query_points<<"\n";
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
    vector<double>* query_projected;


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

int binary_search (double ** reference, double query, int begin, int end)
{
    int length = end - begin+1;
    int end_orig = end;
    int middle_index = (begin + end) / 2;
    double middle = reference[(int)((begin + end) / 2)][point_dim];
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
        double diff1 = abs(query - middle);
        double diff2;
        double diff3;
        if (middle_index < end_orig)
        {
            diff2 = abs(query - reference[(middle_index+1)][point_dim]);
        }
        else {
            diff2 =numeric_limits<double>::max() ;
        }
        if (middle_index > 0)
        {
            diff3 = abs(query - reference[middle_index-1][point_dim]);
        }
        else
        {
            diff3 = numeric_limits<double>::max();
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

spliting_result binary_search_split(vector<double> *input, int start_index, int end_index, double query)
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

spliting_result binary_search_split_(vector<vector<double>> *input, int start_index, int end_index, double query)
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



void exact_knn_projected(vector<vector<int>>* output,const Frame* reference,vector<double>query, double query_projected, int nearest_index, int K, int row, int num_ref_points)
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

    double max_dist = calc_distance(reference->data[start_knn], query, Euclidean);
    num_calc_dis++;
    double dist;
    int calculated_distances_num = 0;
    priority_queue<pair<double, int>> knn;
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
}

spliting_state one_step_parallel_binary_search(vector<double>* query_projected,int start_reference, int end_reference, int start_query, int end_query, vector<vector<int>>* result, Frame* reference, Frame* query, vector<int>* query_order, int k, int num_ref_points)
{
    
    int middle_index = (start_reference + end_reference)/2;
    double middle_value = reference->data[middle_index][point_dim];
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
    vector<double> * query_projected = data->query_projected;
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
            //int binary_search (vector<vector<double>>* reference, double query, int begin, int end)
            //cout<<endl<<"Line:508 : q: "<<q;
         exact_knn_projected     (result                     ,reference,query->row_data[(*query_order)[q]],(*query_projected)[q],nearest, k,q,num_ref_points);       
         //void exact_knn_projected(vector<vector<int>>* output,const Frame* reference,vector<double>query, double query_projected, int nearest_index, int K, int row, int num_ref_points)
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

void print_vector_2D_double (vector<vector<double>>input){
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
    Frame reference("reformed_dataset/0_gr.bin");
    //cout<<"frame one read"<<endl;
    Frame query("reformed_dataset/1_gr.bin", 64);
    //cout<<endl<<query.row_data[3][0]<<endl;
    cout<<query.row_data[3][2];
    int num_ref_points = reference.num_points;
    int num_query_points = query.num_points;
    int num_query_points_orig = num_query_points;
    int round_size = fix_round_size;
    int round_num = num_query_points/round_size;
       vector<double> reference_projected(num_ref_points);
    vector<double> query_projected(num_query_points);
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
    double sum;
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
    //print_double_2d(reference.data, reference.num_points, point_dim+1);
    vector<int> query_order(num_query_points);
    iota(query_order.begin(),query_order.end(),0); //Initializing
    //sort( query_order.begin(),query_order.end(), [&](int i,int j){return query_projected[i]<query_projected[j];} );
    //sort( query_projected.begin(),query_projected.end());
    
    int K_test = 1;
    int num_temp_tets = 64;
    vector<vector<int>> result_test  (num_temp_tets , vector<int> (K_test, 0));
    int score = 0;
    int num_calc_dis_test = 0;
    double avg_test_time = 0;
    double avg_ex_time = 0;
    double test_time;
    //exit(0);

    for (int q= 0 ; q< num_temp_tets;q++)
    {
        //cout<<endl<<"test_number: "<<q<<endl;    
        int nearest_index = binary_search (reference.data,query_projected[q], 0, num_ref_points);
        exact_knn_projected(&result_test,&reference,query.row_data[query_order[q]],query_projected[q], nearest_index, K_test, q,num_ref_points);
    }


exit(0)            ;


        //inja

    
    //print_vector_2D(result_test);
    avg_test_time/= num_temp_tets;
    avg_ex_time/= num_temp_tets;
    cout<<endl<<" test done with score "<<score<<endl<<" rate of cutoff: "<<(num_calc_dis_test/num_temp_tets)/(double)(num_ref_points)<<endl<<"number of calculated distances: "<<num_calc_dis_test<<endl<<"average euc num: "<<num_calc_dis_test/num_temp_tets<<endl<<" num tests: "<<num_temp_tets<<endl<<"avg test time: "<<avg_test_time<<endl<<"avg_ex_time: "<<avg_ex_time<<endl<<"speedup: "<<avg_ex_time/avg_test_time<<endl<<" "<<1/((num_calc_dis_test/num_temp_tets)/(double)(num_ref_points)) <<endl;
    



}