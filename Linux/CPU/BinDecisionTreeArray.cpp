#include<stdio.h>
#include<malloc.h>
#include <string>
#include <iostream>
#include "header.h"
using namespace std;

struct Treenode {
    int index; //index to compare, -1 if leaf
    float data; //value to compare if not leaf, return value if leaf
    bool isLeaf; //leaf or not
    int left; //left child index in tree array
    int right; //right child index in tree array
    string comSymbol; //symbol to compare, "?" for leaf
};

float testDataOnBinDecisionTree(float * dataTableElement, Treenode* treenodes)
{
    int i = 0;
    while(treenodes[i].isLeaf == 0){
        float inputValue = dataTableElement[treenodes[i].index];
        if(inputValue <= treenodes[i].data){
            i = treenodes[i].left;
        }else{
            i = treenodes[i].right;
        } 
    }
    return treenodes[i].data;
}

int main(int argc, const char *argv[])
{
    // Treenode treenodes[31];
    // treenodes[0] = {7,0.052,false,1,2,"<="};

    // treenodes[1] = {20,16.54,false,3,4,"<="};
    // treenodes[2] = {26,0.225,false,5,6,"<="};

    // treenodes[3] = {13,37.61,false,7,8,"<="};
    // treenodes[4] = {21,20.22,false,9,10,"<="};

    // treenodes[5] = {-1,2.0,true,-1,-1,"?"};
    // treenodes[6] = {23,710.2,false,11,12,"<="};

    // treenodes[7] = {21,33.27,false,13,14,"<="};
    // treenodes[8] = {4,0.091,false,15,16,"<="};

    // treenodes[9] = {-1,2.0,true,-1,-1,"?"};
    // treenodes[10] = {17,0.011,false,17,18,"<="};

    // treenodes[11] = {21,25.95,false,19,20,"<="};
    // treenodes[12] = {1,14.12,false,21,22,"<="};

    // treenodes[13] = {-1,2.0,true,-1,-1,"?"};
    // treenodes[14] = {21,34.14,false,23,24,"<="};

    // treenodes[15] = {-1,2.0,true,-1,-1,"?"};
    // treenodes[16] = {17,0.012,false,25,26,"<="};

    // treenodes[17] = {-1,1.0,true,-1,-1,"?"};
    // treenodes[18] = {-1,2.0,true,-1,-1,"?"};

    // treenodes[19] = {-1,2.0,true,-1,-1,"?"};
    // treenodes[20] = {9,0.065,false,27,28,"<="};

    // treenodes[21] = {25,0.361,false,29,30,"<="};
    // treenodes[22] = {-1,1.0,true,-1,-1,"?"};

    // treenodes[23] = {-1,1.0,true,-1,-1,"?"};
    // treenodes[24] = {-1,2.0,true,-1,-1,"?"};

    // treenodes[25] = {-1,2.0,true,-1,-1,"?"};
    // treenodes[26] = {-1,1.0,true,-1,-1,"?"};

    // treenodes[27] = {-1,2.0,true,-1,-1,"?"};
    // treenodes[28] = {-1,1.0,true,-1,-1,"?"};

    // treenodes[29] = {-1,1.0,true,-1,-1,"?"};
    // treenodes[30] = {-1,2.0,true,-1,-1,"?"};

    Treenode treenodes[31];
    treenodes[0] = {22,105.95,false,1,2};

    treenodes[1] = {27,0.159,false,3,4};
    treenodes[2] = {7,0.049,false,5,6};

    treenodes[3] = {13,91.55,false,7,8};
    treenodes[4] = {21,23.47,false,9,10};

    treenodes[5] = {17,0.01,false,11,12};
    treenodes[6] = {26,0.216,false,13,14};

    treenodes[7] = {10,0.643,false,15,16};
    treenodes[8] = {-1,1.0,true,-1,-1};

    treenodes[9] = {-1,2.0,true,-1,-1};
    treenodes[10] = {-1,1.0,true,-1,-1};

    treenodes[11] = {24,0.123,false,17,18};
    treenodes[12] = {-1,2.0,true,-1,-1};

    treenodes[13] = {-1,2.0,true,-1,-1};
    treenodes[14] = {1,15.375,false,19,20};
    
    treenodes[15] = {9,0.054,false,21,22};
    treenodes[16] = {13,49.48,false,23,24};

    treenodes[17] = {8,0.179,false,25,26};
    treenodes[18] = {-1,1.0,true,-1,-1};

    treenodes[19] = {20,17.205,false,27,28};
    treenodes[20] = {-1,1.0,true,-1,-1};

    treenodes[21] = {29,0.065,false,29,30};
    treenodes[22] = {-1,2.0,true,-1,-1};

    treenodes[23] = {-1,1.0,true,-1,-1};
    treenodes[24] = {-1,2.0,true,-1,-1};

    treenodes[25] = {-1,2.0,true,-1,-1};
    treenodes[26] = {-1,1.0,true,-1,-1};

    treenodes[27] = {-1,2.0,true,-1,-1};
    treenodes[28] = {-1,1.0,true,-1,-1};

    treenodes[29] = {-1,2.0,true,-1,-1};
    treenodes[30] = {-1,1.0,true,-1,-1};

    clock_t start,end;
    msd mresult;
    mresult["B"] = 2.0;
    mresult["M"] = 1.0;
	ifstream inputFile;// Input file stream
	string singleInstance;// Single line read from the input file 
	vvs dataTable;// Input data in the form of a vector of vector of strings
    vvd dataTableDouble;// Input data in the form of a vector of vector of Doubles
    inputFile.clear();
    inputFile.open(argv[1]); // Open test file
    if (!inputFile) // Exit if test file is not found
    {
        cerr << "Error: Testing data file not found!" << endl;
        exit(-1);
    }
    while (getline(inputFile, singleInstance)) // Store test data in a table
    {
        parse(singleInstance, dataTable);
    }
    int row = dataTable.size()-1;
    int column = dataTable[0].size()-1;

    const int no_of_input = 569;
    const int no_of_columns = 32;

    int simulate_blocks = 100000;

    float* dataArrayDouble = new float[simulate_blocks * no_of_input * no_of_columns];
    // Stores the predicted class labels for each row in Int
    vd predictedClassLabelsDouble;
    // Stores the given class labels in the test data in Int
    vd givenClassLabelsDouble;
    // Store given class labels in vector of strings named givenClassLabelsDouble
    // Transfer input data from string to Int using map
    for (int i = 1; i < dataTable.size(); i++)
    {
        string data = dataTable[i][1];
        float dataDouble = mresult[data];
        givenClassLabelsDouble.push_back(dataDouble);
        for (int j = 0; j < dataTable[0].size()-1-2; j++){
            dataArrayDouble[(i-1)*32 + j] = std::stod(dataTable[i][j+2]);
        }
    }
    for (int i = 0; i < simulate_blocks; i++)
    {
        for (int j = 0; j < 569 * 32; j++)
        {
            dataArrayDouble[i * 569 * 32 + j] = dataArrayDouble[j];
        }
    }

    start=clock();
    // Predict class labels based on the decision tree
    for(int simulate_ = 0 ; simulate_ < simulate_blocks; simulate_++)
    {
        for (int i = 0; i < row; i++)
        {
            float someDouble = testDataOnBinDecisionTree(dataArrayDouble + simulate_*569*32 + i*32,treenodes);
            predictedClassLabelsDouble.push_back(someDouble);
        }
    }
    
    end=clock();
    cout << "Time Using: " <<(float)(end-start)/CLOCKS_PER_SEC << endl;
	dataTable.clear();
    // Print output
	ofstream outputFile;
	outputFile.open("decisionTreeOutput.txt", ios::app);
	outputFile << endl << "--------------------------------------------------" << endl;
	// Print predictions
    printPredictions(givenClassLabelsDouble, predictedClassLabelsDouble);
	return 0;
}