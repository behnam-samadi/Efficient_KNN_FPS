#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include<cmath>
#include <pthread.h>
using namespace std;




void print_vector (vector<float> v){
    for (int i = 0 ; i< v.size();i++)
    {
        cout<<endl<<i<<" ";
    }
    cout<<endl;
}


float calc_distance (vector<float> v1, vector<float> v2, string type)
{
    cout<<"calc_distance" << v1.size() <<v2.size()<<endl;
    print_vector(v1);
    print_vector(v2);

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


class Frame{
    public:
    int num_points;
    int points_dim;
    vector<vector<float>> data;
};
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

struct thread_data
{
    vector<float> query_point;
    Frame *reference;
    int start_reference;
    int end_reference;
    vector<float> * result;
};





void * call_thread (void* args)
{

    thread_data * args_c = (thread_data*)args;
    
    for (int i = args_c->start_reference; i<args_c->end_reference;i++ )
    {
         float res = calc_distance(args_c->query_point, args_c->reference->data[i],"Manhattan");
         (*(args_c->result))[i] = res;
    }
    cout<<"the thread with data" << args_c->start_reference <<" "<<args_c->end_reference<<"wrote the result"<<endl;
}



void print_frame (Frame frame, int num_points){
    cout<<"it has size:" << frame.data.size()<<" "<<frame.data[0].size()<<endl;
    for (int i = 0; i< num_points;i++)
    {
        for(int j = 0; j<frame.data[0].size();j++)
        {
            cout<<frame.data[i][j]<<" ";
        }
        cout<<endl;
    }

}


void print_vector_2D (vector<vector<float>>input){
    for (int i = 0; i< input.size();i++)
    {
        for(int j = 0; j<input[0].size();j++)
        {
            cout<<input[i][j]<<" ";
        }
        cout<<endl;
    }

}

vector<float> calc_distance_multi_thread (vector<float> query_point, Frame *reference)
{
    int num_threads = 12;
    pthread_t threads[num_threads];
    thread_data data_for_threads[num_threads];
    int slice_size = floor(reference->data.size()/num_threads) ;
    vector<float> result;
    for (int t = 0 ; t<num_threads;t++)
    {
    data_for_threads[t].query_point = query_point;
    data_for_threads[t].reference = reference;
    data_for_threads[t].start_reference = t * slice_size;
    data_for_threads[t].end_reference = (t+1) * slice_size;
    data_for_threads[t].result = &result;
    }
    data_for_threads[num_threads].end_reference = reference->data.size();

    for(int t = 0 ; t<num_threads;t++)
    {
        pthread_create(&(threads[t]), NULL, call_thread, (void*)(&(data_for_threads[t])));
    }
    


    cout<<"joining threads"<<endl;

    for(int t = 0; t<num_threads;t++)
    {
        pthread_join(threads[t], NULL);
    }
    cout<<"threads joined";

    return(*(data_for_threads[0].result));


}

int main(){
    int frame_channels = 3;
    Frame reference = read_data("0000000000.bin", 4, frame_channels);
    Frame query = read_data("0000000001.bin", 4, frame_channels);
    const clock_t begin_time = clock();
    int num_ref_points = reference.data.size();
    int num_query_points = 256;
    cout<< num_ref_points<<" " << num_query_points<<endl;

    for(int i = 0; i<num_query_points;i++)
    {
        vector<float> result = calc_distance_multi_thread(query.data[i],&reference);
        for (int c =0; c<result.size();c++)
        {
            cout<<endl<<result[c]<<" ";
        }
        cout<<endl;
    }
    
return (0);
}
