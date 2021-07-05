#include<stdio.h>
#include<malloc.h>
#include <string>
#include <iostream>
#include "header.h"
using namespace std;

struct Treenode {
    int index; //index to compare, -1 if leaf
    double data; //value to compare if not leaf, return value if leaf
    bool isLeaf; //leaf or not
    int left; //left child index in tree array
    int right; //right child index in tree array
    string comSymbol; //symbol to compare, "?" for leaf
};

double testDataOnBinDecisionTree(double * dataTableElement, Treenode* treenodes)
{
    int i = 0;
    while(treenodes[i].isLeaf == 0){
        double inputValue = dataTableElement[treenodes[i].index];
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
    Treenode treenodes[31];
    treenodes[0] = {7,0.052,false,1,2,"<="};

    treenodes[1] = {20,16.54,false,3,4,"<="};
    treenodes[2] = {26,0.225,false,5,6,"<="};

    treenodes[3] = {13,37.61,false,7,8,"<="};
    treenodes[4] = {21,20.22,false,9,10,"<="};

    treenodes[5] = {-1,2.0,true,-1,-1,"?"};
    treenodes[6] = {23,710.2,false,11,12,"<="};

    treenodes[7] = {21,33.27,false,13,14,"<="};
    treenodes[8] = {4,0.091,false,15,16,"<="};

    treenodes[9] = {-1,2.0,true,-1,-1,"?"};
    treenodes[10] = {17,0.011,false,17,18,"<="};

    treenodes[11] = {21,25.95,false,19,20,"<="};
    treenodes[12] = {1,14.12,false,21,22,"<="};

    treenodes[13] = {-1,2.0,true,-1,-1,"?"};
    treenodes[14] = {21,34.14,false,23,24,"<="};

    treenodes[15] = {-1,2.0,true,-1,-1,"?"};
    treenodes[16] = {17,0.012,false,25,26,"<="};

    treenodes[17] = {-1,1.0,true,-1,-1,"?"};
    treenodes[18] = {-1,2.0,true,-1,-1,"?"};

    treenodes[19] = {-1,2.0,true,-1,-1,"?"};
    treenodes[20] = {9,0.065,false,27,28,"<="};

    treenodes[21] = {25,0.361,false,29,30,"<="};
    treenodes[22] = {-1,1.0,true,-1,-1,"?"};

    treenodes[23] = {-1,1.0,true,-1,-1,"?"};
    treenodes[24] = {-1,2.0,true,-1,-1,"?"};

    treenodes[25] = {-1,2.0,true,-1,-1,"?"};
    treenodes[26] = {-1,1.0,true,-1,-1,"?"};

    treenodes[27] = {-1,2.0,true,-1,-1,"?"};
    treenodes[28] = {-1,1.0,true,-1,-1,"?"};

    treenodes[29] = {-1,1.0,true,-1,-1,"?"};
    treenodes[30] = {-1,2.0,true,-1,-1,"?"};

    clock_t start,end;
    msd mresult;
    mresult["B"] = 1.0;
    mresult["M"] = 2.0;
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
    //double dataArrayDouble[row][column];

    double **dataArrayDouble = new double*[row];
    for(int i = 0; i < row; ++i) {
        dataArrayDouble[i] = new double[column];
    }
    // Stores the predicted class labels for each row in Int
    vd predictedClassLabelsDouble;
    // Stores the given class labels in the test data in Int
    vd givenClassLabelsDouble;
    // Store given class labels in vector of strings named givenClassLabelsDouble
    // Transfer input data from string to Int using map
    for (int i = 1; i < dataTable.size(); i++)
    {
        string data = dataTable[i][1];
        double dataDouble = mresult[data];
        givenClassLabelsDouble.push_back(dataDouble);
        for (int j = 0; j < dataTable[0].size()-1-2; j++){
            dataArrayDouble[i-1][j] = std::stod(dataTable[i][j+2]);
        }
    }
    start=clock();
    // Predict class labels based on the decision tree
    for (int i = 0; i < row; i++)
    {
        double someDouble = testDataOnBinDecisionTree(dataArrayDouble[i],treenodes);
        predictedClassLabelsDouble.push_back(someDouble);
    }
    end=clock();
    cout << "Time Using: " <<(double)(end-start)/CLOCKS_PER_SEC << endl;
	dataTable.clear();
    // Print output
	ofstream outputFile;
	outputFile.open("decisionTreeOutput.txt", ios::app);
	outputFile << endl << "--------------------------------------------------" << endl;
	// Print predictions
    printPredictions(givenClassLabelsDouble, predictedClassLabelsDouble);
	return 0;
}