using namespace std;
#include <string>


extern double** new_2d_mat(int row, int col);
extern void load_feature_matrix(const char* train_feature, double**x);
extern double* double_array(int row);
extern double **normalization(double ** x,int row,int col);
extern void split(int train_row, int train_col, double** train_data, double** train_x,double* train_y);
//extern void predict(vector<LogisticRegression>n models, int train_col, int train_row, int num_classes, double* train_y, \
//double** normalized_x, double** pred_train, int& train_correct);