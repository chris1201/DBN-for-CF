#include "srbm.h"
#include <iomanip>
#include <numeric>
#include <thread>
#include <cmath>
#include <fstream>
#include <iostream>
Srbm::Srbm(const MatrixXi& data, int nHidden, int nVisible)
{
    init(data, nHidden, nVisible);
}

void Srbm::init(const MatrixXi& data, int nHidden, int nVisible)
{
    W.setRandom(nVisible,nHidden);
    hBias.setZero(nHidden);
    vBias.setZero(nVisible);
    this->nHidden=nHidden;
    this->nVisible=nVisible;
    //cout<<W<<hBias<<vBias;
}
void Srbm::expandBias(int n)
{
    hBias_use.setZero(n, nHidden);
    vBias_use.setZero(n, nVisible);
    for(int i=0; i< n; ++i)
    {
        hBias_use.row(i)=hBias;
        vBias_use.row(i)=vBias;
    }
}
void Srbm::train(const MatrixXi& data, float Wlr, float HBiaslr, float Vbiaslr)
{
    int nSample=data.rows();
    expandBias(nSample);
    cout<<"hBias_use\n"<<hBias_use<<'\n'<<"vBias_use\n"<<vBias_use<<endl;
}
