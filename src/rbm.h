#ifndef RBM_H
#define RBM_H

#include "rbmdata.h"
#include "rbmpredictdata.h"
#include <Eigen/Dense>
#include <vector>
#include <thread>

using namespace Eigen;
using namespace std;

typedef Matrix<bool, 1, Dynamic> RowVectorXb;

class Rbm {
public:
    Rbm(const RbmData& data, int nHidden);

    virtual void performEpoch(const RbmData& data);

    virtual double predict(
            const RbmData& data, 
            const RbmPredictData& predictData, 
            const string& filename);

    virtual double predict(
            const RbmData& data,
            const RbmPredictData& predictData);

    virtual double predict(
            const RbmData& data,
            const RbmPredictData& predictData,
            ostream& predictStream);

    static float hBiasLearnRate;
    static float vBiasLearnRate;
    static float WlearnRate;
    static float weightDecay;
    static float momentum;
    static int T;
private:
    Rbm();

    void negActivation(const RowVectorXf& h0states);

    void gibbsSample();

    void normalizedNegActivation(const RowVectorXf& h0states);
    void softmax(const RowVectorXf& h0states);

    void initVisibleBias(const RbmData& data);

    void applyMomentum(const RbmData& data, int user);
    void selectWeights(const RbmData& data, int rangeStart, int rangeEnd);

public: // These are public so RbmParallel can use them
    MatrixXf W, Wsel, Wmomentum;
    RowVectorXf vBias, vBiasSel, vBiasMomentum, hBias, hBiasMomentum;

    void performEpoch(const RbmData& data, int userStart, int userEnd);

private:
    RowVectorXf h0probs, hTProbs;
    RowVectorXb h0states;

    RowVectorXf negData;

    MatrixXf posProds, negProds;

    int nHidden;
    int nClasses;
};

#endif
