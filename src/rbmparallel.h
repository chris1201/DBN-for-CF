#ifndef RBM_PARALLEL_H
#define RBM_PARALLEL_H
#include "rbm.h"
#include "rbmdata.h"

class RbmParallel : public Rbm {
public:
    RbmParallel(int nThreads, const RbmData& data, int nHidden);

    void performEpoch(const RbmData& data);

private:
    RbmParallel();

    void startEpochs(const RbmData& data, int batchStart, int batchSize);
    void joinExecution();
    void updateWeights();
    void synchronizeWeights();

    static void performEpochInThread(
            Rbm& r, const RbmData& data, int userStart, int userEnd);

    int nThreads;
    int nUsers;
    vector<Rbm> subRbms;
    vector<thread> threads;
};

#endif
