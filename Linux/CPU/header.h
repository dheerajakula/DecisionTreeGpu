#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <math.h>
#include <float.h>
#include <cstdlib>
#include <iomanip>
#include <time.h>
#include <stdio.h>
#include <memory.h>
#include <sys/mman.h>
#include <dlfcn.h>

using namespace std;

typedef vector<string> vs;
typedef vector<vs> vvs;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<double> vd;
typedef vector<vd> vvd;
typedef map<string, int> msi;
typedef map<string, double> msd;
typedef map<double, int> mdi;
typedef int (* rules_func_t)(int * a);

// struct node defines the structure of a node of the decision tree
struct node
{
	int attribute; //Stores the attribute which is used to compare during the inference
	double splitOn;// Stores which attribute to split on at a particular node during the training
	double label;// Stores the class label for leaf nodes.
    /*For nodes that are not leaf nodes, it stores the value of the attribute of the parent's' split*/
	bool isLeaf;// boolean flag for leaf nodes
	vector<double> childrenValues;// Store the values of the childrens' attributes
	vector<node*> children;// Store pointers to the children of a node
};

void parse(string&, vvs&);// Parse a single line from the input file and stores the information into a vector of vector of strings
void printDecisionTree(node* nodePtr);// Print tree
vvd pruneTable(vvd&, double&, double);// Prune a table based on a column/attribute's name and the value of that attribute. Removes that column and all instances that have that value for that column
node* buildDecisionTree(vvd&, node*, vvd&);// Build the decision tree based on the table it is passed
bool isHomogeneous(vvd&);// Return true if all instances in a subtable at a node have the same class label
vi countDistinct(vvd&, int);// Return a vector of integers containing the counts of all the various values of an attribute/column
double decideSplittingColumn(vvd&);// Return the column on which to split on. Decision of column is based on entropy
inline int returnColumnIndex(double&, vvd&);// Return the index of a column in a subtable
bool tableIsEmpty(vvd&);// Return true if a subtable is empty
double testDataOnDecisionTree(vd&, node*/*, vvd&*/);// Run a single instance of the test data through the decision tree and return the predicted class label
inline int returnIndexOfVector(vd&, double);// Return the index of a string in a vector of strings
void printPredictions(vd&, vd&);// Output the predictions to file
vvd generateTableInfo(vvd &dataTable);// Generate information about the table in a vector of vector of stings
void changeDecisionTree(node* nodePtr, int number);
/*inline int returnColIndex(double columnName, double tableInfoItem, int index){
	return (columnName == tableInfoItem) ? index : -1;
}*/