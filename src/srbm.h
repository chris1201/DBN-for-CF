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
    void train(const MatrixXi& data, float Wlr, float HBiaslr, float Vbiaslr);
    void expandBias(int n);
    MatrixXf W;
    RowVectorXf hBias, vBias;
    MatrixXf hBias_use, vBias_use;
    int nHidden, nVisible;
    float Wlr, hBiaslr, vbiaslr;
    MatrixXi X,Y;

private:
    MatrixXf h0probs, hTProbs;
    MatrixXi h0states, negData;
    MatrixXf posProds, negProds;
};
