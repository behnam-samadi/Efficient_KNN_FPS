class Frame{
    //later: change to private
public:
    vector<vector<float>> data;
    Frame(string file_adress, int max_points = 0)
    {

    ifstream fin(file_adress, ios::binary);
    fin.seekg(0, ios::end);
    const size_t num_elements = fin.tellg() / sizeof(double);
    cout<<endl<<num_elements<<endl<<sizeof(double);
    fin.seekg(0, ios::beg);
    exit(0);
    fin.read(reinterpret_cast<char*>(&data_temp[0]), num_elements*sizeof(float))
    



    //ifstream fin(file_adress);
    
    
    bool finished = false;

    string whole = "";
    string temp = "";
    char current;
    getline(fin, whole, 'e');
    int frame_size = 0;
    for (int c = 0; c<whole.length(); c++)
    {
        if (whole[c] == ',')
            frame_size++;
    }
    frame_size /= Points_Dim;
    cout<<endl<<frame_size;
    //int frame_size = whole.length();
    if (max_points!=0)
    {
        frame_size = max_points;
    }
    cout<<endl<<frame_size;
    
    //cout<<a1;
    //cout<<" "<<a1.length();
    //exit(0);
    int point_num = 0;
    int dim_read = 0;
    vector<float> temp_point(Points_Dim);
    int counter=0;
    while(point_num<frame_size)
    {

        //cout<<a1<<" has been read" ;
        current = whole[counter];
        if (current!=',')
        {
            temp = temp+ current;
        }
        else
        {

            
            temp_point[dim_read++] = stof(temp);
            temp = "";
            if (dim_read == Points_Dim)
            {
                point_num++;
                if (point_num%100 == 0) cout<<point_num<<"'s point has been read"<<endl;
                data.push_back(temp_point);
                dim_read=0;
            }
        }
        /*//getline(fin, a1, 'e');      
        if (temp!='e')
        {
            if (temp != ',')
            {

            }
            //cout<<stof(a1)<<" has been read"<<endl ;
            counter++;
            if ((counter%100) == 0)
            {
                cout<<counter<<endl;
            }
            //cout<<stof(a1)<<" has been read"<<endl ;
            data.push_back(vector<float>(point_dim));       
            data[data.size()-1][0] = stof(a1);
            for (int c = 1 ;c<point_dim;c++)
            {
                getline(fin, a1, ',');      
                data[data.size()-1][c] = stof(a1);
            }
        }
        else finished = true;*/
        
    counter++;
    }
    }
};


