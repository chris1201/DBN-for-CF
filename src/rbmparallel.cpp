#include "rbmparallel.h"

RbmParallel::RbmParallel(int nThreads, const RbmData& data, int nHidden) : 
    Rbm(data, nHidden), 
    nThreads(nThreads), 
    nUsers(data.range.size() - 1),
    subRbms(nThreads - 1, Rbm(data, nHidden))
{
    synchronizeWeights();
}

void RbmParallel::performEpoch(const RbmData& data) {
    int batchSize = nUsers;
    for (int batchStart = 0; batchStart < nUsers; batchStart += batchSize) {
        startEpochs(data, batchStart, batchSize);
        joinExecution();
        updateWeights();
    }
}

void RbmParallel::startEpochs(
        const RbmData& data, int batchStart, int batchSize) {
    threads.clear();
    int usersPerStep = batchSize / nThreads;
    int userStart = batchStart;
    for (int i = 0; i < nThreads - 1; i++) {
        threads.push_back(
                thread(&RbmParallel::performEpochInThread, 
                    ref(subRbms[i]), ref(data), 
                    userStart, userStart + usersPerStep));
        userStart += usersPerStep;
    }
    Rbm::performEpoch(data, userStart, nUsers);
}

void RbmParallel::performEpochInThread(
        Rbm& r, const RbmData& data, int userStart, int userEnd) {
    r.performEpoch(data, userStart, userEnd);
}

void RbmParallel::joinExecution() {
    for (auto it = threads.begin(); it != threads.end(); it++) {
        it->join();
    }
}

void RbmParallel::updateWeights() {
    float updateFactor = 1.0 / nThreads;
    W *= updateFactor;
    vBias *= updateFactor;
    hBias *= updateFactor;
    for (auto it = subRbms.begin(); it != subRbms.end(); it++) {
        W += it->W * updateFactor;
        vBias += it->vBias * updateFactor;
        hBias += it->hBias * updateFactor;
    }
    synchronizeWeights();
}

void RbmParallel::synchronizeWeights() {
    for (auto it = subRbms.begin(); it != subRbms.end(); it++) {
        it->W = W;
        it->hBias = hBias;
        it->vBias = vBias;
    }
}
