#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
using namespace std;




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

vector <vector<int>> KNN (Frame frame1, Frame frame2, int K){
    int num_ref_points = frame1.data.size();
    int num_query_points = frame2.data.size();
    vector<vector<int>> result  (num_query_points , vector<float> (K, 0));
    vector<float>  distance (num_ref_points);

    for(int i = 0; i<num_query_points;i++){
        distance[i] = 
    }

}


int main(){
    
    Frame frame1 = read_data("0000000000.bin", 4, 3);
    Frame frame2 = read_data("0000000001.bin", 4, 3);
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
    
    //cout<<frame1.num_points<<" "<<frame1.points_dim<<"\n";
    //cout<<"\n" << frame2.num_points<<" "<<frame2.points_dim;
    
return (0);
}
