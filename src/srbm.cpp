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
void Srbm::train(MatrixXi data, float Wlr, float HBiaslr, float Vbiaslr)
{
    MatrixXf X=data.cast<float>();
    MatrixXf visData=X;
    int nSample=data.rows();
    expandBias(nSample);
    h0probs = 1 / (1 + (-X*W - hBias_use).array().exp());

    h0states = (h0probs.array() >
    (h0probs.Random(h0probs.rows(),h0probs.cols()).array() + 1) / 2).cast<float>();
    negActivation(h0states);
    hTProbs = 1 / (1 + (-negData*W - hBias_use).array().exp());

    posProds.noalias() = visData.transpose() * h0probs;
    negProds.noalias() = negData.transpose() * hTProbs;
    W+=Wlr*(posProds - negProds);
    cout<<"posProds\n"<<posProds<<"\nnegProds\n"<<negProds<<endl;
    cout<<"negData\n"<<negData<<endl;
}
void Srbm::negActivation(MatrixXf h0states)
{
    negData=1/(1+(-h0states*W.transpose()-vBias_use).array().exp());
    //negData=
    //negData.array() = negData.array().exp();
}
