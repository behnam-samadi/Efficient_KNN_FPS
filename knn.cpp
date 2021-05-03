#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include<cmath>
#include <pthread.h>
using namespace std;




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


float calc_distance (vector<float> v1, vector<float> v2, string type)
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


class Frame{
    public:
    int num_points;
    int points_dim;
    vector<vector<float>> data;
};

struct thread_data
{
    Frame *query;
    Frame *reference;
    int K;
    int start_query;
    int end_query;
    vector<vector<int>> result;
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



void print_vector_2D_int (vector<vector<int>>input){
    cout<<"prinintg" <<"size"<<input.size()<<" "<<input[0].size();
    for (int i = 0; i< input.size();i++)
    {
        for(int j = 0; j<input[0].size();j++)
        {
            cout<<input[i][j]<<" ";
        }
        cout<<endl;
    }

return;}



vector<vector<int>> KNN (Frame * reference, Frame * query, int K, int query_start, int query_end){
    cout<<"one thread is working";
    int num_ref_points = reference->data.size();
    //int num_query_points = query->data.size();
    int num_query_points = query_end - query_start;
    cout<<"num_query_points"<<num_query_points<<endl;
    vector<vector<int>> result  (num_query_points , vector<int> (K, 0));
    vector<float>  distance (num_ref_points);
    for(int i = query_start; i<query_end;i++){
        //cout<<" i is " << i << "from" << query_start <<" "<< query_end<<endl;
        //cout<<"KNN, Progress:" <<(float)i/num_query_points<<"\n";
        for (int j = 0; j<num_ref_points;j++)
        {
            distance[j] = calc_distance(query->data[i], reference->data[j], "Manhattan");
        }
        vector<int> topk = topK(distance, K);
        //cout<<"top was selected"<<endl;
        for(int c = 0; c<K;c++)
        {
            result[i-query_start][c] = topk[c];
        }

    }
    cout<<"one thread is returning";
    cout<<"/////////**********////////------------"<<endl<<endl;
    print_vector_2D_int(result);
    cout<<"/////////**********////////------------"<<endl<<endl;

return(result);
}

void * call_thread (void* args){
    thread_data arguments = *(thread_data*) args;

    //cout<<"one thread is running";
    //cout<<endl<<"one thread is writing result**********************************";
    ((thread_data*)args)->result = KNN(arguments.reference, arguments.query, arguments.K, arguments.start_query, arguments.end_query);
    //cout<<endl<<"------------\n---\n---*******--one thred wrote the result\n-----\n---***********----";
    

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


void print_vector_2D (vector<vector<float>>input, int num_points){
    for (int i = 0; i< num_points;i++)
    {
        for(int j = 0; j<input[0].size();j++)
        {
            cout<<input[i][j]<<" ";
        }
        cout<<endl;
    }

}


int main(){
    int frame_channels = 3;
    Frame reference = read_data("0000000000.bin", 4, frame_channels);
    Frame query = read_data("0000000001.bin", 4, frame_channels);
    const clock_t begin_time = clock();

    int number_of_ref_points = 256;
    int number_of_nearest_point = 20;
    int num_threads = 2;


    int slice_size = number_of_ref_points / num_threads;
    pthread_t threads[num_threads];
    thread_data data_for_threads[num_threads];
    cout<<"preparing data"<<endl;
    for (int t= 0; t < num_threads;t++)
    {
        data_for_threads[t].reference = &reference;
        data_for_threads[t].query = &query;
        data_for_threads[t].K = number_of_nearest_point;
        data_for_threads[t].start_query = (t*slice_size);
        data_for_threads[t].end_query = (t+1) * slice_size;
        
    }



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

    //for(int  t = 0; t<num_threads;t++)
    //{
      //  cout<<data_for_threads[t].result[0][0];
    //}
        float run_time = float(clock() - begin_time);
    cout<<endl<< "run_time" <<run_time;
    cout<<"asle dastan:"<<endl;
    cout<<data_for_threads[0].result.size();
    //print_vector_2D_int(data_for_threads[0].result);
    cout<<endl;
cout<<"finishied";
//cout<<endl<<endl<<"az inja";
    //int FN = 1;
    //for (int j =0; j<slice_size;j++) for (int c = 0; c<frame_channels;c++)cout<<" "<<slices[FN].data[j][c]<<" ,";
        //cout<<endl<<"az inja2:";
    //for (int j =0; j<slice_size;j++) for (int c = 0; c<frame_channels;c++)cout<<" "<<frame2.data[FN*slice_size+ j][c]<<" ,";
    //cout<<endl<<slices[2].data;

    //cout<<knn.size()<<"\n";
    //cout<<knn[0].size()<<"\n";
    //cout<<"inja:"<<"\n"<<"\n";
    //cout<<endl;
    //for (int i = 0; i<knn.size();i++){
        //for (int j = 0; j< knn[0].size();j++){
        //cout<< knn[i][j]<<"\n";}
        //cout<<endl;

    //}
    
    
    //cout<<frame1.num_points<<" "<<frame1.points_dim<<"\n";
    //cout<<"\n" << frame2.num_points<<" "<<frame2.points_dim;
    
return (0);
}
