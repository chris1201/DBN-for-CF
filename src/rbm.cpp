#include "rbm.h"
#include <iomanip>
#include <numeric>
#include <thread>
#include <cmath>
#include <fstream>

float Rbm::hBiasLearnRate = 0.001;
float Rbm::vBiasLearnRate = 0.008;
float Rbm::WlearnRate = 0.0006;
float Rbm::weightDecay = 0.0001;
float Rbm::momentum = 0.5;
int Rbm::T = 1;

Rbm::Rbm(const RbmData& data, int nHidden) 
        : nHidden(nHidden), nClasses(data.nClasses)
{
    W.setRandom(data.nMovies * nClasses, nHidden);
    W.array() *= 0.01;
    vBias.setZero(data.nMovies * nClasses);
    initVisibleBias(data);
    hBias.setZero(nHidden);
}

void Rbm::initVisibleBias(const RbmData& data) {
    MatrixXi totals = MatrixXi::Zero(1, data.nMovies);
    int nUsers = data.range.size() - 1;
    for (int i = 0; i < nUsers; i++) {
        const auto& movies = data.movies.segment(
                data.range(i), data.range(i + 1) - data.range(i));
        const auto& ratings = data.ratings.segment(
                data.range(i), data.range(i + 1) - data.range(i));
        int amountOfRatings = data.range(i + 1) - data.range(i);
        for (int r = 0; r < amountOfRatings; r++) {
            vBias(movies(r)) += ratings(r);
            int exactMovie = movies(r) / nClasses;
            totals(exactMovie) += ratings(r);
        }
    }
    for (int m = 0; m < totals.size(); m++) {
        for (int c = 0; c < nClasses; c++) {
            if (vBias(m * nClasses + c) != 0) {
                vBias(m * nClasses + c) /= totals(m);
                vBias(m * nClasses + c) = 
                    log(vBias(m * nClasses + c));
            }
        }
    }
}

void Rbm::performEpoch(const RbmData& data) {
    performEpoch(data, 0, data.range.size() - 1);
}

void Rbm::performEpoch(const RbmData& data, int userStart, int userEnd) {
    vector<int> randomizedIds(userEnd - userStart, 0);
    for (int i = userStart; i < userEnd; i++) 
        randomizedIds[i - userStart] = i;
    random_shuffle(randomizedIds.begin(), randomizedIds.end());

    for (unsigned i = 0; i < randomizedIds.size(); i++) {
        int rangeStart = data.range(randomizedIds[i]);
        int rangeEnd = data.range(randomizedIds[i] + 1); // end is exclusive
        int rangeLength = rangeEnd - rangeStart;
        
        selectWeights(data, rangeStart, rangeEnd);
        const auto& visData = data.ratings.segment(rangeStart, rangeLength);

        h0probs = 1 / (1 + (-visData*Wsel - hBias).array().exp());
        h0states = h0probs.array() > 
            (h0probs.Random(h0probs.size()).array() + 1) / 2;
        negActivation(h0states.cast<float>());

        hTProbs = 1 / (1 + (-negData*Wsel - hBias).array().exp());
        for (int t = 1; t < T; t++)
            gibbsSample();

        posProds.noalias() = visData.transpose() * h0probs;
        negProds.noalias() = negData.transpose() * hTProbs; 

        if (i > 0)
            applyMomentum(data, randomizedIds[i - 1]);

        hBiasMomentum.noalias() = hBiasLearnRate * (h0probs - hTProbs);
        hBias.noalias() += hBiasMomentum;
        vBiasMomentum.noalias() = vBiasLearnRate * (visData - negData);
        for (int r = rangeStart; r < rangeEnd; r++)
            vBias(data.movies(r)) += vBiasMomentum(r - rangeStart);

        Wmomentum.noalias() = posProds - negProds;
        for (int r = 0; r < rangeLength; r++)
            W.row(data.movies(r + rangeStart)).noalias() += 
                WlearnRate * (Wmomentum.row(r) - weightDecay*Wsel.row(r));
    }
}

void Rbm::negActivation(const RowVectorXf& h0states) {
    softmax(h0states);
}

void Rbm::gibbsSample() {
    h0states = 
        hTProbs.array() > (hTProbs.Random(hTProbs.size()).array() + 1) / 2;
    negActivation(h0states.cast<float>());
    hTProbs = 1 / (1 + (-negData*Wsel - hBias).array().exp());
}

void Rbm::normalizedNegActivation(const RowVectorXf& h0states) {
    softmax(h0states);
}

void Rbm::softmax(const RowVectorXf& h0states) {
    negData = (h0states*Wsel.transpose() + vBiasSel);
    for (int m = 0; m < negData.size(); m += nClasses) {
        negData.segment(m, nClasses).array() -=
            negData.segment(m, nClasses).maxCoeff();
    }
    negData.array() = negData.array().exp();
    for (int m = 0; m < negData.size(); m += nClasses) {
        negData.segment(m, nClasses).array() /=
            negData.segment(m, nClasses).sum();
    }
}

void Rbm::selectWeights(const RbmData& data, int rangeStart, int rangeEnd) {
    int rangeLength = rangeEnd - rangeStart;
    Wsel.resize(rangeLength, nHidden);
    vBiasSel.resize(rangeLength);
    for (int r = rangeStart; r < rangeEnd; r++) {
        Wsel.row(r - rangeStart).noalias() = W.row(data.movies(r));
        vBiasSel(r - rangeStart) = vBias(data.movies(r));
    }
}

void Rbm::applyMomentum(const RbmData& data, int user) {
    if (momentum == 0.0) return;
    int rangeStart = data.range(user);
    int rangeEnd = data.range(user + 1); // end is exclusive
    int rangeLength = rangeEnd - rangeStart;

    hBias.noalias() += momentum * hBiasMomentum;
    for (int r = rangeStart; r < rangeEnd; r++)
        vBias(data.movies(r)) += momentum * vBiasMomentum(r - rangeStart);

    for (int r = 0; r < rangeLength; r++) {
        W.row(data.movies(r + rangeStart)).noalias() += 
            momentum * Wmomentum.row(r);
    }
}

double Rbm::predict(const RbmData& data, const RbmPredictData& predictData,
        const string& fname) {
    ofstream out(fname.c_str());
    double d = predict(data, predictData, out);
    out.close();
    return d;
}

double Rbm::predict(const RbmData& data, const RbmPredictData& predictData) {
    stringstream dontcare;
    return predict(data, predictData, dontcare);
}

double Rbm::predict(const RbmData& data, const RbmPredictData& predictData,
        ostream& predictStream) {
    double rmse = 0.0;
    int predictCount = 0;
    for (unsigned i = 0; i < predictData.userIds.size(); i++) {
        int userId = predictData.userIds[i];
        int rangeStart = data.range(userId);
        int rangeEnd = data.range(userId + 1); // end is exclusive
        selectWeights(data, rangeStart, rangeEnd);
        const auto& visData = 
            data.ratings.segment(rangeStart, rangeEnd - rangeStart);
        h0probs = 1 / (1 + (-visData*Wsel - hBias).array().exp());

        rangeStart = predictData.range(i);
        rangeEnd = predictData.range(i + 1);
        selectWeights(predictData, rangeStart, rangeEnd);
        normalizedNegActivation(h0probs);
        const auto& actualData = 
            predictData.ratings.segment(rangeStart, rangeEnd - rangeStart);

        for (int r = 0; r < actualData.size(); r += nClasses) {
            float actual = 0;
            float predicted = 0;
            for (int c = 0; c < nClasses; c++) {
                actual += actualData(r + c) * (c + 1);
                predicted += negData(r + c) * (c + 1);
            }
            predictStream << predicted << endl;
            float t = actual - predicted;
            rmse += t * t;
        }
        predictCount += actualData.size() / nClasses;
    }
    return sqrt(rmse / predictCount);
}
