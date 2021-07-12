
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <stdio.h>
#include <ctime>
#include <string>
#include <fstream>
#include <map>

// Parses a string and stores data into a vector of vector of strings
void parse(std::string& someString, std::vector<std::vector<std::string>>& attributeTable)
{
    int attributeCount = 0;
    std::vector<std::string> vectorOfStrings;
    while (someString.length() != 0 && someString.find(',') != std::string::npos)
    {
        size_t pos;
        std::string singleAttribute;
        pos = someString.find_first_of(',');
        singleAttribute = someString.substr(0, pos);
        vectorOfStrings.push_back(singleAttribute);
        someString.erase(0, pos + 1);
    }
    vectorOfStrings.push_back(someString);
    attributeTable.push_back(vectorOfStrings);
    vectorOfStrings.clear();
}

// class node to store a binary tree node
class Node
{
public:
    int attr_index;
    float attr_value;
    bool isleaf;
    int l;
public:
    __host__ __device__
    Node()
    {
        attr_index = 0;
        attr_value = 0;
        isleaf = false;
        l = 0;
    }
    __host__ __device__
    Node(int index, float value, bool leaf, int lef)
    {
        attr_index = index;
        attr_value = value;
        isleaf = leaf;
        l = lef;
    }
    __host__ __device__
    bool is_leaf()
    {
        return isleaf;
    }
    __host__ __device__
    bool def_left()
    {
        return true;
    }
    __host__ __device__
    int fid()
    {
        return attr_index;
    }
    __host__ __device__
    int left(int val)
    {
        return l;
    }
    __host__ __device__
    float thresh()
    {
        return attr_value;
    }
};

// class tree to store a array of binary tree nodes. Has the rensponsibility to transfer the nodes to gpu
class Tree
{
public:
    Node* host_node_array;
    Node* dev_node_array;
    int size;
    __host__ __device__
    Tree(Node* node_array, int s)
    {
        host_node_array = node_array;
        dev_node_array = 0;
        size = s;
    }

    __device__
    Node& operator[](int index)
    {
        assert(index < size);
        return dev_node_array[index];
    }

    __host__
    void AssignNodesToGpu()
    {
        cudaError_t cudaStatus = cudaMalloc((void**)&dev_node_array, size * sizeof(Node));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
        }
        cudaStatus = cudaMemcpy(dev_node_array, host_node_array, size * sizeof(Node), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
        }

    }

};

// device code to infer one tree
__device__ float InferOneTree(Tree tree, const float* input)
{
    int curr = 0;
    int count = 0;
 
    for (;;) {
        Node n = tree[curr];
        if (n.is_leaf()) break;
        float val = input[n.fid()];
        bool cond = isnan(val) ? !n.def_left() : val >= n.thresh();
        curr = n.left(curr) + cond;
        count++;
    }

    float out = tree[curr].thresh();

    return out;
}

// kernel to infer one tree
__global__ void MySingleTreeKernel(Tree* tree, float* input, int columns, float* dev_output)
{
    int no_of_columns = columns;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // slice the input
    float* input_slice = input + i * no_of_columns;
    float output = InferOneTree(*tree, input_slice);
    //printf("output %f", output);
    dev_output[i] = output;
}

Tree* InitializeModel()
{
    const int treeSize = 31;
    Node treenodes[treeSize];

    // create a binary decision tree.
    treenodes[0] = Node(7, 0.052, false, 1);

    treenodes[1] = Node(20, 16.54, false, 3);
    treenodes[2] = Node(26, 0.225, false, 5);

    treenodes[3] = Node(13, 37.61, false, 7);
    treenodes[4] = Node(21, 20.22, false, 9);

    treenodes[5] = Node(-1, 2.0, true, -1);
    treenodes[6] = Node(23, 710.2, false, 11);

    treenodes[7] = Node(21, 33.27, false, 13);
    treenodes[8] = Node(4, 0.091, false, 15);

    treenodes[9] = Node(-1, 2.0, true, -1);
    treenodes[10] = Node(17, 0.011, false, 17);

    treenodes[11] = Node(21, 25.95, false, 19);
    treenodes[12] = Node(1, 14.12, false, 21);

    treenodes[13] = Node(-1, 2.0, true, -1);
    treenodes[14] = Node(21, 34.14, false, 23);

    treenodes[15] = Node(-1, 2.0, true, -1);
    treenodes[16] = Node(17, 0.012, false, 25);

    treenodes[17] = Node(-1, 1.0, true, -1);
    treenodes[18] = Node(-1, 2.0, true, -1);

    treenodes[19] = Node(-1, 2.0, true, -1);
    treenodes[20] = Node(9, 0.065, false, 27);

    treenodes[21] = Node(25, 0.361, false, 29);
    treenodes[22] = Node(-1, 1.0, true, -1);

    treenodes[23] = Node(-1, 1.0, true, -1);
    treenodes[24] = Node(-1, 2.0, true, -1);

    treenodes[25] = Node(-1, 2.0, true, -1);
    treenodes[26] = Node(-1, 1.0, true, -1);

    treenodes[27] = Node(-1, 2.0, true, -1);
    treenodes[28] = Node(-1, 1.0, true, -1);

    treenodes[29] = Node(-1, 1.0, true, -1);
    treenodes[30] = Node(-1, 2.0, true, -1);

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        fprintf(stderr, "failed intializing model!");
    }

    Tree my_tree = Tree(treenodes, treeSize);

    my_tree.AssignNodesToGpu();

    Tree* dev_my_tree;

    // Assign memory to device tree
    cudaStatus = cudaMalloc((void**)&dev_my_tree, sizeof(Tree));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        fprintf(stderr, "failed intializing model!");
    }

    // Copy tree from host memory to GPU.
    cudaStatus = cudaMemcpy(dev_my_tree, &my_tree, sizeof(Tree), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        fprintf(stderr, "failed intializing model!");
    }

    return dev_my_tree;
}

float* LoadRequest_to_host(int simulate_blocks, int no_of_input, int no_of_columns)
{
    using namespace std;
    // read input file
    map<string, float> mresult;
    mresult["B"] = 1.0;
    mresult["M"] = 2.0;

    ifstream inputFile;// Input file stream
    string singleInstance;// Single line read from the input file 
    vector<vector<string>> dataTable;// Input data in the form of a vector of vector of strings
    vector<vector<float>> dataTableDouble;// Input data in the form of a vector of vector of floats
    inputFile.clear();
    inputFile.open("data.csv"); // Open test file
    if (!inputFile) // Exit if test file is not found
    {
        cerr << "Error: Testing data file not found!" << endl;
        exit(-1);
    }
    while (getline(inputFile, singleInstance)) // Store test data in a table
    {
        parse(singleInstance, dataTable);
    }

    int row = dataTable.size() - 1;
    int column = dataTable[0].size() - 1;

    

    float* dataArrayFloat = new float[simulate_blocks * no_of_input * no_of_columns];



    // Stores the predicted class labels for each row in Int
    vector<float> predictedClassLabelsfloat;
    // Stores the given class labels in the test data in Int
    vector<float> givenClassLabelsfloat;

    // Store given class labels in vector of strings named givenClassLabelsDouble
    // Transfer input data from string to Int using map
    for (int i = 1; i < dataTable.size(); i++)
    {
        string data = dataTable[i][1];
        float dataFloat = mresult[data];
        givenClassLabelsfloat.push_back(dataFloat);
        for (int j = 2; j < dataTable[0].size() - 1; j++) {
            dataArrayFloat[(i - 1) * 32 + j - 2] = std::stof(dataTable[i][j]);
        }
    }

    for (int i = 0; i < simulate_blocks; i++)
    {
        for (int j = 0; j < 569 * 32; j++)
        {
            dataArrayFloat[i * 569 * 32 + j] = dataArrayFloat[j];
        }
    }

    return dataArrayFloat;
}

float* LoadRequest_to_device(int simulate_blocks, int no_of_input, int no_of_columns, float* dataArrayFloat)
{
    cudaError_t cudaStatus;
    float* dev_input = 0;
    cudaStatus = cudaMalloc((void**)&dev_input, simulate_blocks * no_of_columns * no_of_input * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_input, dataArrayFloat, simulate_blocks * no_of_columns * no_of_input * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    return dev_input;
}

float* assignMem_to_output(int simulate_blocks, int no_of_input, int no_of_columns)
{
    cudaError_t cudaStatus;
    float* dev_output = 0;
    cudaStatus = cudaMalloc((void**)&dev_output, simulate_blocks * no_of_input * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    return dev_output;
}

int main()
{
    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };*/
    using namespace std;

    cudaError_t cudaStatus;
    Tree* dev_my_tree;
    dev_my_tree = InitializeModel();

    

    int simulate_blocks = 10000;
    const int no_of_input = 569;
    const int no_of_columns = 32;

    float* dataArrayFloat = LoadRequest_to_host(simulate_blocks,no_of_input, no_of_columns);

    clock_t begin = clock();

    float* dev_input = LoadRequest_to_device(simulate_blocks, no_of_input, no_of_columns, dataArrayFloat);
    
    float* dev_output = assignMem_to_output(simulate_blocks, no_of_input, no_of_columns);

    float* host_output = new float[simulate_blocks*no_of_input];

    cudaEvent_t start, stop;
    float elapsedTime;
    /*cudaEventCreate(&start);
    cudaEventRecord(start, 0);*/

    MySingleTreeKernel << <simulate_blocks, 569 >> >(dev_my_tree, dev_input, no_of_columns, dev_output);

    /*cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed kernel time : %f ms\n", elapsedTime);*/

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
       
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
    }

    cudaStatus = cudaMemcpy(host_output, dev_output, simulate_blocks * no_of_input * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("Elapsed CPU time : %f ms\n", elapsed_secs*1000);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


