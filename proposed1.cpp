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

class Frame{
    public:
    int num_points;
    int points_dim;
    vector<vector<float>> data;
};

class parallel_search_result{
    vector <int>  values;
    vector <bool> inits;
    public:
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
        //cout<<"sum become:"<<sum<<"\n";
    }
    //cout<<"\n"<<"sum:"<<sum;

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
        //cout<<"maxarg: " <<min_arg<<"\n";
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
    //imidiate
    //num_ref_points = 12;
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
    //cout<<middle_index<<endl;
    //cout<<middle<<endl;
    //exit(0);
    while (end >= begin)
    {
        middle_index = (begin + end) / 2;
        middle = (*reference)[(int)((begin + end) / 2)];
        //cout<<"begin and end"<<begin<<" "<<end<<endl;
        if (query == middle) 
        {
            //cout<<"inja";
            return (middle_index);
        }
        else if (query > middle) 
        {
            //cout<<"begin and end in loop"<<begin<<" "<<end<<endl;
            //cout<<"inja1 ";
            //cout<<middle_index<<" ";
            //cout<<begin<<endl;
            begin = middle_index+1;
        }
        else if(query < middle) 
            {
                //cout<<"inja2";
                end = middle_index-1;
            }
        }
        //cout<<"unsuccesfull"<<endl;
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
        cout<<diff1<<" "<<diff2<<" "<<diff3<<endl;
        if ((diff1 <= diff2) && (diff1 <= diff3))  {
        cout<<"first"<<endl;
        return(middle_index);
        }
        else if ((diff2 <= diff1) && (diff2 <= diff3))
        {
            cout<<"second"<<endl;
            return(middle_index+1);
        }
        else if((diff3 <= diff2) && (diff3 <= diff1)) 
        {
        cout<<"third"<<endl;
        return(middle_index-1);
        }

    
}

/*void * thread_function(void *input)
{
    thread_data data = *((thread_data *)input);
    if (data.points.size() == 1)
    {

    }
}*/

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
};

spliting_result binary_search_split(vector<float> *input, int start_index, int end_index, float query)
{
    cout<<endl<<"one call to binary_search_split with values " <<start_index<<" "<< end_index<<" "<<query;
    int start_orig = start_index;
    int end_orig = end_index;
    bool successful = 0;
    int middle_index;
    while (end_index >= start_index)
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
    spliting_result result;
    if (!successful)
    {
        int divide_point = start_index;
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
    cout<< "one_step_parallel_binary_search is called with values"<<start_reference<<" "<<end_reference<<" "<<start_query<<" "<<end_query<<endl;
    int middle_index = (start_reference + end_reference)/2;
    int middle_value = (*reference)[middle_index];
    spliting_result split = binary_search_split(query, start_query, end_query, middle_value);
    int divider1 = split.divider1;
    int divider2 = split.divider2;
    cout<<endl<<"before set_value"<<endl;
    // assigning result to the found points
    result->set_value((*reference)[divider1], divider1, divider2);
    cout<<endl<<"after set_value"<<endl;
    spliting_state state;
    state.left_size = abs(divider1 - start_query);
    state.middle_size = divider2 - divider1;
    state.right_size = abs(divider2 - end_query);
    state.divider1 = divider1;
    state.divider2 = divider2;
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
    cout<<"one call to main function with values"<<start_reference<<" "<< end_reference<<" "<< start_query<<" "<<end_query<<endl;
    if ((end_reference - start_reference) < 5)
    {
        cout<<"running parallel_binary_search with reference size smaller than 5"<<endl;
        for (int q = 0 ; q<= end_query;q++)
        {
            int binary_search (vector<float>* reference, float query, int begin, int end);
            int single_result = binary_search(reference, (*query)[q], start_reference, end_reference);
            result->set_value(single_result, q, q+1);
        }
    }
    int middle_index = (start_reference + end_reference)/2;
    spliting_state state;
    do{
    cout<<"calling one_step_parallel_binary_search"<<endl;
    state = one_step_parallel_binary_search(reference , query , start_reference, end_reference , start_query, end_query, result);
    cout<<"successful call to one_step_parallel_binary_search"<<endl;
        if(state.right_size > 0)
        {
            pthread_t* temp = new pthread_t;           
            cout<<endl<<"thread cretaed"<<endl;
            threads->push_back(temp);
            thread_data* args;
            args->reference = reference;
            args->query = query;
            args->start_reference = middle_index + 1;
            args->end_reference = end_reference;
            args->start_query = state.divider2;
            args->end_query = end_query;
            args->result = result;
            args->threads = threads;
            cout<<"starting creating thread with valus"<<args->start_reference<<" "<<args->end_reference<<" "<<args->start_query<<" "<<args->end_query<<endl;
            pthread_create(temp, NULL, parallel_binary_search, (void*)&args);
        }
    } while(state.left_size>1);
    if (state.left_size == 1)
        {
            int signle_result = binary_search(reference, (*query)[start_query], start_query, middle_index -1);
            result->set_value((*reference)[signle_result], start_query, start_query+1);
        }
}
int main()
{
	int frame_channels = 3;
    Frame reference = read_data("0000000000.bin", 4, frame_channels);
    Frame query = read_data("0000000001.bin", 4, frame_channels);
    int num_ref_points = reference.data.size();
    int num_query_points = query.data.size();
    num_query_points = 512;
    int round_size = 64;

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



    vector<int> sorted_query(round_size);
    iota(sorted_query.begin(),sorted_query.end(),0); //Initializing
    sort( sorted_query.begin(),sorted_query.end(), [&](int i,int j){return sum_cordinates_query[i]<sum_cordinates_query[j];} );
    sort( sum_cordinates_query.begin(),sum_cordinates_query.end());


    //print_vector(sum_cordinates);
    //print_vector_float(sum_cordinates);
    //cout<<"and:"<<endl;
    //print_vector_float(sum_cordinates_query);
    //cout<<endl<<sorted_indices.size();
    /*for (int i = 0; i< 64; i++)
    {
        cout<<query.data[i][0]<<", "<<query.data[i][1]<<" ,"<<query.data[i][2]<<endl;
    }*/

    thread_data args;
    parallel_search_result round_result (round_size);
    vector<pthread_t*> round_threads;
    args.reference = &sum_cordinates;
    args.query = &sum_cordinates_query;
    args.start_reference = 0;
    args.end_reference = reference.data.size();
    args.start_query = 0;
    args.end_query = round_size;
    args.result = &round_result;
    args.threads = &round_threads;
    cout<<endl<<endl<<endl<<"start running parallel code"<<endl;
    parallel_binary_search((void*)(&args));

    while(!(round_result.all_set()))
        {cout<<"waiting";}
    for(int t = 0; t<round_threads.size();t++)
    {
        pthread_join(*(round_threads[t]),NULL);
    }


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









    return(0);
    int num_round = num_query_points / round_size;
    for (int round = 0 ; round < num_round;round++)
    {
        
    }

    exit(0);
    int k = 40;
    vector<vector<int>> knn_result = KNN(reference, query, k, "Modified_Manhattan",num_query_points);
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