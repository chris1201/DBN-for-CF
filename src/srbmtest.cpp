#include "srbm.h"
#include <iostream>
using namespace Eigen;
using namespace std;
int main()
{
    MatrixXi data;
    data.setZero(2,5);
    data<<1,0,0,0,0,
          0,1,0,0,0;
    MatrixXf W;
    W.setRandom(5,3);
    //cout<<data.cast<float>()*W;
    //cout<<"data\n"<<data<<endl;
    Srbm rbm(data, 3, 5);
    rbm.train(data,1,1,1);

}
