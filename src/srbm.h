#include <Eigen/Dense>
#include <vector>
#include <thread>

using namespace Eigen;
using namespace std;
typedef Matrix<bool, 1, Dynamic> RowVectorXb;
class Srbm
{
public:
    Srbm(const MatrixXi& data, int nHidden, int nVisible);
    void init(const MatrixXi& data, int nHidden, int nVisible);
    void train(MatrixXi data, float Wlr, float HBiaslr, float VBiaslr);
    void expandBias(int n);
    void negActivation(MatrixXf h0states);
    MatrixXf W;
    RowVectorXf hBias, vBias;
    MatrixXf hBias_use, vBias_use;
    int nHidden, nVisible;
    float Wlr, hBiaslr, vbiaslr;
    MatrixXi X,Y;

private:
    MatrixXf h0probs, hTProbs;
    MatrixXf h0states, negData;
    MatrixXf posProds, negProds;
};
