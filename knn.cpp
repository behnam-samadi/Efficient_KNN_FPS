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
    int num_ref;
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

vector<vector<int>> KNN (Frame reference, Frame query, int K, int num_query=0){
    cout<<"one thread is working";
    int num_ref_points = reference.data.size();
    int num_query_points = query.data.size();
    if (!(num_query == 0)) num_query_points = num_query;
    vector<vector<int>> result  (num_query_points , vector<int> (K, 0));
    vector<float>  distance (num_ref_points);
    for(int i = 0; i<num_query_points;i++){
        //cout<<"KNN, Progress:" <<(float)i/num_query_points<<"\n";
        for (int j = 0; j<num_ref_points;j++)
        {
            distance[j] = calc_distance(query.data[i], reference.data[j], "Manhattan");
        }
        vector<int> topk = topK(distance, K);
        for(int c = 0; c<K;c++)
        {
            result[i][c] = topk[c];
        }

    }
    cout<<"one thread is reurnin";
return(result);
}

void * call_thread (void* args){
    
    thread_data arguments = *((thread_data*) args);
    
    Frame * a = arguments.reference;
    Frame * b = arguments.query;
    int c = arguments.K;
    int d = arguments.num_ref;
    cout<<"one thread is running";
    vector<vector<int>> knn = KNN(*(arguments.reference), *(arguments.query), arguments.K, arguments.num_ref);
    cout<<endl<<"one thread is writing result";
    ((thread_data*)args)->result = knn;
    cout<<endl<<"-----------------one thred wrote the result";
    return (0);

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
    Frame frame1 = read_data("0000000000.bin", 4, frame_channels);
    Frame frame2 = read_data("0000000001.bin", 4, frame_channels);
    cout<<"two frames created\n";
    int i = frame1.data.size();
    int j = frame1.data[0].size();
    //return (0);
    cout<<i<<j<<"\n";
    for (int m = 0;m<5;m++)
    {
        for (int n = 0 ; n<j;n++)
        {
            cout<<frame1.data[m][n]<<" ";
        }
        cout<<"\n";
    }
    std::vector<float> v1 = {1,2,3};
    std::vector<float> v2 = {4,6,3};
    //cout<<"\n"<<"distance: "<<calc_distance(v1,v2, "Manhattan");
    const clock_t begin = clock();
    int number_of_ref_points = 64;
    int number_of_nearest_point = 20;
    vector<vector<int>> knn = KNN(frame1, frame2, number_of_nearest_point, number_of_ref_points);
    float run_time = float(clock() - begin);
    cout<<endl<< "run_time" <<run_time;
    int num_threads = 4;
    //print_frame(frame2, 12);
    

    //slicing the data;
    Frame slices[num_threads];
    int slice_size = number_of_ref_points / num_threads;
    for(int i = 0; i< num_threads;i++)
    {
        vector<vector<float>> temp  (slice_size , vector<float> (frame_channels, 0));
        for (int j = 0; j < slice_size; j++)
        {
            for (int c = 0; c<frame_channels;c++){
            temp[j][c] = frame2.data[i*slice_size + j][c];
        }
        }
        slices[i].data = temp;
        slices[i].num_points = slice_size;
        slices[i].points_dim = frame_channels;
    }
    print_frame(slices[2], slices[2].data.size());


    pthread_t threads[num_threads];
    thread_data data_for_threads[num_threads];
    cout<<"preparing data"<<endl;
    for (int t= 0; t < num_threads;t++)
    {
        data_for_threads[t].reference = &frame1;
        data_for_threads[t].query = &(slices[t]);
        //imidiate
        data_for_threads[t].K = number_of_nearest_point;
        data_for_threads[t].num_ref = number_of_ref_points;
    }
    cout<<"------"<<endl;
    print_frame(*(data_for_threads[2].query), 16);
    cout<<"and---"<<endl;
    print_vector_2D(slices[2].data, 16);
    cout<<"----------";
    exit(0);
    cout<<"creating threads"<<endl;

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

    for(int  t = 0; t<num_threads;t++)
    {
        cout<<data_for_threads[t].result[0][0];
    }
    cout<<"asle dastan:"<<endl;
    //print_vector_2D(data_for_threads[0].result, 12);
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
