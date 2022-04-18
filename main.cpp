#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <memory>
#include <cmath>
#include <vector>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <map>
#include "utility.h"
#include "lr.h"
// g++ -Wall -std=c++11 main.cpp utility.cpp -o main

using namespace std;

// to be updated
// ------------------------------------------------------
void usage(const char* prog){

   cout << "Read training data then classify test data using logistic regression:\nUsage:\n" << prog << " [options] [training_data]" << endl << endl;
   cout << "Options:" << endl;  
   cout << "-t <file>  training dataset" << endl; 
   cout << "-p <file>  test dataset" << endl; 
   cout << "-s <int>   Shuffle dataset after each iteration. default 1" << endl;    
   cout << "-i <int>   Maximum iterations. default 50000" << endl;   
   cout << "-e <float> Convergence rate. default 0.005" << endl;    
   cout << "-a <float> Learning rate. default 0.001" << endl; 
   cout << "-l <float> L1 regularization weight. default 0.0001" << endl; 
   // cout << "-m <file>  Read weights from file" << endl;    
  // cout << "-o <file>  Write weights to file" << endl;        
   cout << "-r         Randomise weights between -1 and 1, otherwise 0" << endl;    
   cout << "-v         Verbose." << endl << endl;      
}
//-----------------------------------------------------------------------

int main(int argc, const char* argv[]){

    // Learning rate
    double alpha = 0.1;
    // L1 penalty weight
    double l1 = 0.0001;
    // Max iterations
    unsigned int maxit = 10;
    // Shuffle data set
    int shuf = 1;
    // Convergence threshold
    double eps = 0.005;
    // Verbose
    int verbose = 0;
    // Randomise weights
    int randw = 0;
    // Read model file
    string model_in = "";
    // Write model file
    string model_out = "";
    const char* train_file = "";    // train file 
    const char* test_file = "";  // test file
    int train_row = 0; // # of rows in training set
    int train_col = 0; // # columns including label in training set
    int num_classes = 0;
    int test_row = 0; // # of rows in training set
    int test_col = 0; 
    

    if(argc < 2){
        usage(argv[0]);
        return(1);
    }else{
        cout << "# called with:       ";
        for(int i = 0; i < argc; i++){
            cout << argv[i] << " ";
            if(string(argv[i]) == "-t" && i < argc-1){
                train_file = argv[i+1];
            }
            if(string(argv[i]) == "-nl" && i < argc-1){
                num_classes = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-p" && i < argc-1){
                test_file = argv[i+1];
            }
            if(string(argv[i]) == "-nr" && i < argc-1){
                train_row = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-nc" && i < argc-1){
                train_col = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-a" && i < argc-1){
                alpha = atof(argv[i+1]);
            }
            if(string(argv[i]) == "-tr" && i < argc-1){
                test_row = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-tl" && i < argc-1){
                test_col = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-i" && i < argc-1){
                maxit = atoi(argv[i+1]);
            }
            
            if(string(argv[i]) == "-l" && i < argc-1){
                l1 = atof(argv[i+1]);
            }
            if(string(argv[i]) == "-h"){
                usage(argv[0]);
                return(1);
            }
        }
        cout << endl;
    }
    vector<LogisticRegression> models(num_classes, train_col);
    double **train_x = new_2d_mat(train_row, train_col); // holding features of training 
    double *train_y = double_array(train_row); // holding labels of training
    double **train_data = new_2d_mat(train_row, train_col+1);  
    load_feature_matrix(train_file, train_data);
    int correct = 0; // correct label for training set
    double **pred_train = new_2d_mat(train_row, num_classes); 
    double **test_x = new_2d_mat(test_row, test_col);  
    double *test_y = double_array(test_row); 
    double **test_data = new_2d_mat(test_row, test_col+1);
    load_feature_matrix(test_file, test_data);
    double **pred_test = new_2d_mat(test_row, num_classes);
    int test_correct = 0;
    
    // split training's features and labels
    split(train_row, train_col,train_data,train_x,train_y);
    split(test_row, test_col,test_data,test_x,test_y);
    
    // normalization 
    double **normalized_x = new_2d_mat(train_row, train_col);
    double **normalized_test_x = new_2d_mat(test_row, test_col);
    normalized_x = normalization(train_x, train_row, train_col);
    normalized_test_x = normalization(test_x, test_row, test_col);

    for (int i = 0; i < num_classes; i++)
        {
            LogisticRegression model(train_col);
            double *y_temp = double_array(train_row);
            for (int j = 0; j < train_row; j++)
            {
                if (train_y[j] == i)
                {
                    y_temp[j] = 1;
                }
                else
                    y_temp[j] = 0;
            }
            model.fit(normalized_x,y_temp,train_row, train_col,alpha,l1,maxit);
            cout<<"-----------------"<<endl;
            //cout << i << endl;
            models[i] = model;
        }

    //predict(models, train_col, train_row, num_classes, train_y, normalized_x, pred_train, train_correct);
    //predict(models, test_col, test_row, num_classes, test_y, normalized_test_x, pred_test, test_correct);
    for (int i = 0; i < train_row; i++)
        {
            double max_sigm = 0;
            int pred_label = 0;
            int label = train_y[i];
            for (int j = 0; j < num_classes; j++)
            {
                LogisticRegression model(train_col);
                model = models[j];
                pred_train[i][j] = model.classify(normalized_x[i]);
                if (max_sigm < pred_train[i][j])
                {
                    max_sigm = pred_train[i][j];
                    pred_label = j;
                }
            }
            if (label == pred_label)  correct++;
        }
    cout << "Train Accuracy:" << double(double(correct) / double(train_row)) << endl;
    for (int i = 0; i < test_row; i++)
        {
            double max_sigm = 0;
            int pred_label = 0;
            int label = train_y[i];
            for (int j = 0; j < num_classes; j++)
            {
                LogisticRegression model(test_col);
                model = models[j];
                pred_test[i][j] = model.classify(normalized_test_x[i]);
                if (max_sigm < pred_test[i][j])
                {
                    max_sigm = pred_test[i][j];
                    pred_label = j;
                }
            }
            if (label == pred_label)  test_correct++;
        }
    cout << "Test Accuracy:" << double(double(test_correct) / double(test_row)) << endl;
    //cout<<"-----------------"<<endl;
    // for (long i =0;i<train_row;i++) {
    //     for (long j=0;j<train_col-1;j++) {
    //         cout<< normalized_x[i][j]<<endl;
    //     }
    // }

}