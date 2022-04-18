#include <cmath>
#include <memory>
#include <iostream>
#include <random>
#include "utility.h"
#include "lr.h"

using namespace std;

double dot_product(double* a, double* b, int n) { //static?
    double out = 0.0;
    for (long i = 0; i < n; i++) {
        out += a[i] * b[i];
    }
    return out;
}

LogisticRegression::LogisticRegression(int features)
{
    new_weights = new double[features];
    old_weights = new double[features];
    total_l1 = new double[features];
    features_num = features;
    bias = 0.0;
    bias_old = 0.0;
}

double LogisticRegression::sigmoid(double x){

    static double overflow = 20.0;
    if (x > overflow) x = overflow;
    if (x < -overflow) x = -overflow;
    return 1.0/(1.0 + exp(-x));
}

// generate random number between 0 and 1
double LogisticRegression::rand_generator() {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(0,1);
    return distr(eng);
}

// l2 norm between array a and b
double LogisticRegression::l2norm (double* a, double* b, int n) {
    double out = 0.0;
    for (long i = 0; i < n; i++) {
        out += pow(a[i] - b[i],2);
    }
    return sqrt(out);
}

double LogisticRegression::classify(double* x) {
    return classify_wrapper(x, new_weights, features_num, bias);
}

double LogisticRegression::classify_wrapper(double* x, double* new_weights, int features_num, double bias) {
    double logit = dot_product(x, new_weights, features_num) + bias;
    return sigmoid(logit);
}

// training data(x) and class i(y)
void LogisticRegression::fit(double **x, double *y, int m, int n, double alpha, double l1, int max_iter) {
    
    memset(old_weights, 0, sizeof(double) * features_num);
    memset(new_weights, 0, sizeof(double) * features_num);
    memset(total_l1, 0, sizeof(double) * features_num);

    // initialize weights
    for (int i = 0; i < features_num; i++) old_weights[i] = rand_generator();

    double *predict = new double[m];
    double mu = 0.0;
    double norm = 1.0;
    int iter = 0;

    while (norm > 0.0000001) {
        double loss = 0.0;
        for (int i=0; i<m; i++) {
            mu += l1* alpha; 
            predict[i] = classify_wrapper(x[i], old_weights, features_num, bias_old); //sigmoid
            for (int j=0; j<features_num; j++) {
                double gradient = (y[i]-predict[i])*x[i][j];
                new_weights[j] = old_weights[j] + alpha*gradient;
                if (l1) {
                    double weight_temp = new_weights[j];
                    if(new_weights[j] > 0.0){
                                new_weights[j] = max(0.0,new_weights[j] - (mu + total_l1[j]));
                    } else if(new_weights[i] < 0.0){
                                new_weights[j] = min(0.0,new_weights[j] + (mu - total_l1[j]));
                            }
                    total_l1[j] += (new_weights[j] - weight_temp);
                }
            }
            loss += -((y[i] * log(predict[i]) + (1 - y[i]) * log(1 - predict[i])) / m);
            std::swap(old_weights, new_weights);
        }
        norm = l2norm(new_weights, old_weights, features_num);
        iter++;
        if (iter > max_iter) break;
        cout << "cross_entropy_loss:" << loss << endl;
        
    }
    
}


