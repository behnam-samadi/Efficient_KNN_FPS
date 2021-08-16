#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include<cmath>
#include <pthread.h>
#include "numeric"
#include <limits>
using namespace std;
pthread_mutex_t myMutex;


ofstream fout("output.txt");
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
            this->values[i] = value;
            this->inits[i] = 1;
            cout<<endl<<i<<"'th element is changed to:" << value<<endl;
            //fout<<endl<<i<<"'th element is changed to:" << value<<endl;
            fout<<endl<<i<<"'th element is changed to:" << value<<endl;
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


float calc_distance (vector<float> v1, vector<float> v2, string type)
{
    
    if (type == "Modified_Manhattan")
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
        if (type=="Euclidean")
        sum+= pow(abs(v1[i] - v2[i]), 2);
        if (type=="Manhattan")
        sum+= abs(v1[i] - v2[i]);
    }

    float result = sum;
    if (type == "Euclidean")
        result = sqrt(result);
    return(result);
    }
}


vector<int> topK(vector<float> input, int K){
    float inf = 0;
    for (int i = i ; i< input.size();i++)
    {
        if (input[i] > inf) inf = i;
    }
    inf = inf + 100;

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

vector<vector<int>> KNN (Frame reference, Frame query, int K,string metric,  int num_query=0){
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

void print_vector (vector<int> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<v[i]<<" ";
    }
    cout<<endl;
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


spliting_state one_step_parallel_binary_search(vector<float> *reference, vector<float>* query,int start_reference, int end_reference, int start_query, int end_query, parallel_search_result* result)
{
    

    //cout<< "one_step_parallel_binary_search is called with values"<<start_reference<<" "<<end_reference<<" "<<start_query<<" "<<end_query<<endl;
    int middle_index = (start_reference + end_reference)/2;
    //fout<<"1:"<<middle_index<<endl;
    float middle_value = (*reference)[middle_index];
    spliting_result split = binary_search_split(query, start_query, end_query, middle_value);
    int divider1 = split.divider1;
    int divider2 = split.divider2;
    //int temp_res  = binary_search (query, middle_value, start_query, end_query);
    //divider1 = temp_res;
    //divider2 = temp_res;
    //divider2 = divider1;
    //cout<<endl<<"before set_value"<<endl;
    // assigning result to the found points
    result->set_value(middle_index, divider1, divider2);
    //cout<<endl<<"after set_value"<<endl;
    spliting_state state;
    state.left_size = max(divider1 - start_query, 0);
    state.middle_size = max(divider2 - divider1, 0);
    state.right_size = max(end_query - divider2 + 1, 0);
    state.divider1 = divider1;
    state.divider2 = divider2;
    state.middle_index = middle_index;
    
    //fout<< endl<<(end_query - start_query+1)<<" splitted to: "<<  state.left_size<<" "<<state.middle_size<<" "<< state.right_size<<endl;
    //" "<<state.divider1<<" "<<state.divider2<<endl;
    //cout<<"p returning"<<endl;
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
    parallel_search_result* result = data->result;
    vector<pthread_t*>* threads = data->threads;
    

    //fout<<endl<<"one call to main function with values"<<start_reference<<" "<< end_reference<<" "<< start_query<<" "<<end_query<<endl;
    
    int middle_index;
    spliting_state state;
    int task_size;
    do{
    
        
        
        //fout<<"2:"<<middle_index<<endl;
        if (end_query < start_query)
    {
        //fout<<endl<<"deleted because end is smaller than start"<<endl;
        //pak shod
        delete data;
        return NULL;
    }
    if ((end_reference - start_reference) < 3)
    {
        cout<<"running parallel_binary_search with reference size smaller than 5"<<endl;
        //fout<<"running parallel_binary_search with reference size smaller than 5"<<endl;
        //fout<<"this points are being processed serialy: ";
        for (int q = start_query ; q<= end_query;q++)
        {
            //fout<<q<<"and ";

            //int binary_search (vector<float>* reference, float query, int begin, int end);
            //fout<<"single binary search is called for: " <<(*query)[q]<<" " <<start_reference<<" "<< end_reference;
            int single_result = binary_search(reference, (*query)[q], start_reference, end_reference);
            //fout<<endl<<"smaller than 5 assigning: " <<single_result<<" "<<q<<endl;
            cout<<endl<<"smaller than 5 assigning: " <<single_result<<" "<<q<<endl;
            //fout<<"2"<<" "<<endl;
            result->set_value(single_result, q, q+1);
        }
        //fout<<endl;
        //pak shod
        delete data;
        return NULL;
    }
    //task_size = end_query - start_query + 1;
    //cout<<"now, calling one_step_parallel_binary_search with values"<<start_reference<<" "<<end_reference<<" "<<start_query<<" "<<end_query<<" "<<endl;
    state = one_step_parallel_binary_search(reference , query , start_reference, end_reference , start_query, end_query, result);
    middle_index = state.middle_index;
    //cout<<"successful call to one_step_parallel_binary_search"<<endl;
        if(state.right_size > 0)
        {
            pthread_t* temp = new pthread_t;           
            //cout<<endl<<"thread cretaed"<<endl<<threads<<"and temp: "<<temp<<endl;

            pthread_mutex_lock(&myMutex);

            threads->push_back(temp);

            pthread_mutex_unlock(&myMutex);
            //cout<<"successful threads->push_back(temp)"<<threads->size()<<endl;
            thread_data* args = new thread_data;
            args->reference = reference;
            args->query = query;

            //if (middle_index < end_reference) 
            args->start_reference = middle_index;//fout<<"oomad too asli"<<endl;}

            //else
            //{
                //fout<<"oomad too else"<<endl;
            //    args->start_reference = middle_index;
            //}
            args->end_reference = end_reference;
            args->start_query = state.divider2;
            args->end_query = end_query;
            args->result = result;
            args->threads = threads;
    //fout<<"3:"<<middle_index<<endl;

            if (args->end_query >= args->start_query)
            {
            //fout<<endl<<endl<<"starting creating thread with values"<<args->start_reference<<" "<<args->end_reference<<" "<<args->start_query<<" "<<args->end_query<<" "<<middle_index <<"original "<<start_reference<<" "<<end_reference<< endl;

            //fout<<"creatign thread: reference: from "<<args->start_reference<<"..........."<<args->end_reference<<"and query from "<<args->start_query<<"............."<<args->end_query<<endl;
            pthread_create(temp, NULL, parallel_binary_search, (void*)args);
            }
            else
            {
                //pak shod
                delete args;
            }
        }

        end_query = state.divider1-1;
        end_reference = middle_index;
        //fout<<endl<<endl<<"now left serach is "<<start_reference<<" "<<end_reference<<" "<<start_query<<" "<<end_query<<" with left size " <<state.left_size<<endl<<endl;
    } while(state.left_size>1);
    if (state.left_size == 1)
        {
            //fout<<endl<<start_query<<" is a left point to "<<end_query<<endl;
            //int binary_search (vector<float>* reference, float query, int begin, int end)
            //fout<<endl<<"one point found! "<<start_query<<"to "<<end_query<<endl;
            int single_result = binary_search(reference, (*query)[start_query], start_reference, end_reference);
            //fout<<"1"<<" "<<endl;
            result->set_value(single_result, start_query, start_query+1);
        }
        //pak shod
        delete data;
}
int main()
{
    //test binary_search_split
    vector<float> test = {1, 1 , 1 ,2.1, 2.2, 2.3, 3, 3, 3, 4.4, 5, 6 ,7 , 7, 7 , 7};
    while(0)
    {
        float float_query;
        cout<<"enter the query (the size is "<<test.size()<<")"<<endl;
        cin>> float_query;
    spliting_result test_r =  binary_search_split(&test, 0,test.size()-1 , float_query);
    cout<<"result:"<<endl<<test_r.divider1<<" "<<test_r.divider2<<endl;
    }
    int frame_channels = 3;
    Frame reference = read_data("0000000000.bin", 4, frame_channels);
    Frame query = read_data("0000000001.bin", 4, frame_channels);
    int num_ref_points = reference.data.size();
    int num_query_points = query.data.size();
    num_query_points = 1024;
    int round_size = num_query_points;

    cout<< "num_ref_points: "<< num_ref_points<<"num_query_points: " << num_query_points<<endl;

    

    
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


    int k = 20;
    
    vector<float> test_reference  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40};
    vector<float> test_query = {2.2, 3.5, 21.6, 35.7, 90};
    spliting_result split = binary_search_split(&test_query, 0, 4, 93);
    cout<<split.divider1<<" "<<split.divider2<<" ";

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
    pthread_mutex_init(&myMutex,0);
    parallel_binary_search((void*)(args));

    while(!(round_result.all_set()));
        //{cout<<endl<<"waiting";}
    for(int t = 0; t<round_threads.size();t++)
    {
        pthread_join(*(round_threads[t]),NULL);
    }

    //fout<<endl<<endl<<endl<<"final result"<<endl;
    cout<<endl<<round_threads.size()<<endl;
    for (int p = 0 ; p<round_size;p++)
    {
        cout<<p<<" "<<round_result.values[p]<<endl;
    }
    //print_vector(round_result.values);
    

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






    vector<vector<int>> knn_result = KNN(reference, query, k, "Modified_Manhattan",num_query_points);

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
    vector<vector<int>>  ground_truth = KNN(reference, query, k, "Euclidean",num_query_points);
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

    //vector<float> test2{10, 20 , 30};
    //vector<float> test1{110, 120 , 130};
    //cout<<calc_distance(test1, test2, "Modified_Manhattan");
    //for (int i = 0 ; i<num_ref_points;i++)cout<<reference.data[i][0]<<","<<reference.data[i][1]<<","<<reference.data[i][2]<<endl;
}