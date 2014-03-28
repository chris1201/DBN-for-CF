#include "rbmpredictdata.h"
#include "rbmdata.h"
#include "rbm.h"
#include "rbmparallel.h"
#include <fstream>
#include <string>
#include <cstdlib>
#include <deque>

char helpString[] =
    "Usage: %s [arguments] <trainfile> <testfile>\n"
    "\n"
    "Arguments:\n"
    "  -h                    Print Help (this message) and exit\n"
    "  -v <float>            Set visible bias learning rate to <float>\n"
    "  -H <float>            Set hidden bias learning rate to <float>\n"
    "  -w <float>            Set weight learning rate to <float>\n"
    "  -c <float>            Set weight-cost coefficient to <float>\n"
    "  -n <uint>             Use <uint> hidden nodes\n"
    "  -i <float>            Set initial momentum to <float>\n"
    "  -m <float> <uint>     Set final momentum to <float> at epoch <uint>\n"
    "  -e <uint>             Perform <uint> epochs\n"
    "  -t <uint> [<uint>...] Increase T by one at epoch <uint>, <uint> ...\n"
    "  -d                    Print the defaults and exit\n"
    "  -s <uint>             Set the random seed to <uint>\n"
    "  --save <string>       Save the predictions of the model after epoch\n"
    "         <uint> [...]        <uint> to file: <string>+<uint>.dat\n"
    "  --never               Do not predict, unless specified to save.\n"
    "\n"
    "Example usage:\n"
    "  ./rbm -t 5 5 7 train.dat test.dat\n"
    "     Set T to 3 at epoch 5, and to 4 at epoch 7.\n"
    "  ./rbm --never --save mypredfile 10 20 train.dat test.dat\n"
    "     Generates two files: mypredfile10.dat and mypredfile20.dat\n";

string trainFilename, testFilename, savePrefix;
float vLearn = 0.005, hLearn = 0.005, wLearn = 0.005, wCost = 0.005;
int nHidden = 100, nEpochs = 40;
float finalMomentum = 0.0, initialMomentum = 0.0;
int finalMomentumStart = 5;
deque<int> tIncrements;
deque<int> epochsToSaveAt;
int seed = 1;
bool predictAlways = true;
bool parallel = false;
int nThreads = 32;

void printConfig() {
    cout << "Learning rates:" << endl
         << "  visible        " << vLearn << endl
         << "  hidden         " << hLearn << endl
         << "  weights        " << wLearn << endl
         << "  cost           " << wCost << endl;
    cout << "Hidden nodes:    " << nHidden << endl;
    if (initialMomentum != 0.0 && finalMomentum != 0.0) {
        cout << "Momentum:        " << endl
             << "  initial        " << initialMomentum << endl
             << "  final          " << finalMomentum << endl
             << "  start          " << finalMomentumStart << endl;
    }
    if (tIncrements.size() > 0) {
        cout << "T-increment on:  ";
        for (unsigned i = 0; i < tIncrements.size(); i++) {
            cout << tIncrements[i] << ' ';
        }
        cout << endl;
    }
    if (epochsToSaveAt.size() > 0) {
        cout << "Saving at epochs:";
        for (unsigned i = 0; i < epochsToSaveAt.size(); i++) {
            cout << epochsToSaveAt[i] << ' ';
        }
        cout << endl;
        cout << "Save prefix:     " << savePrefix << endl;
    }
    cout << "Datasets:      " << endl
         << "  train          " << trainFilename << endl
         << "  test           " << testFilename << endl;
    cout << "Epochs:          " << nEpochs << endl;
    cout << "Random seed:     " << seed << endl;
    cout << "Predict always:  " << (predictAlways? "yes" : "no") << endl;
    cout << endl;
}

void parseArgs(int argc, char* argv[]) {
    int current = 1;
    // TODO: make more user friendly in terms of error handling.
    //       and coding style leaves something to be desired...
    while (current < argc) {
        if (strcmp("-h", argv[current]) == 0) {
            printf(helpString, argv[0]);
            exit(0);
        } else if (strcmp("-d", argv[current]) == 0) {
            cout << "Defaults: " << endl;
            printConfig();
            exit(0);
        } else if (strcmp("-v", argv[current]) == 0) {
            vLearn = atof(argv[current + 1]);
            current += 1;
        } else if (strcmp("-H", argv[current]) == 0) {
            hLearn = atof(argv[current + 1]);
            current += 1;
        } else if (strcmp("-w", argv[current]) == 0) {
            wLearn = atof(argv[current + 1]);
            current += 1;
        } else if (strcmp("-n", argv[current]) == 0) {
            nHidden = atoi(argv[current + 1]);
            current += 1;
        } else if (strcmp("-i", argv[current]) == 0) {
            initialMomentum = atof(argv[current + 1]);
            current += 1;
        } else if (strcmp("-m", argv[current]) == 0) {
            finalMomentum = atof(argv[current + 1]);
            finalMomentumStart = atoi(argv[current + 2]);
            current += 2;
        } else if (strcmp("-e", argv[current]) == 0) {
            nEpochs = atoi(argv[current + 1]);
            current += 1;
        } else if (strcmp("-c", argv[current]) == 0) {
            wCost = atof(argv[current + 1]);
            current += 1;
        } else if (strcmp("-t", argv[current]) == 0) {
            do {
                current += 1;
                tIncrements.push_back(atoi(argv[current]));
            } while (current + 1 < argc && 
                    '0' <= argv[current + 1][0] && argv[current + 1][0] <= '9');
        } else if (strcmp("-s", argv[current]) == 0) {
            current += 1;
            seed = atoi(argv[current]);
        } else if (strcmp("--save", argv[current]) == 0) {
            current += 1;
            savePrefix = argv[current];
            do {
                current += 1;
                epochsToSaveAt.push_back(atoi(argv[current]));
            } while (current + 1 < argc && 
                    '0' <= argv[current + 1][0] && argv[current + 1][0] <= '9');
        } else if (strcmp("--never", argv[current]) == 0) {
            predictAlways = false;
        } else if (argc - current == 2) {
            trainFilename = argv[current];
        } else if (argc - current == 1) {
            testFilename = argv[current];
        } else {
            cerr << "ERROR: Unknown option: " << argv[current] << endl
                 << "Exiting now." << endl;
            exit(1);
        }
        current += 1;
    }
}

void safeLoad(RbmData& p, const string& fname) {
    ifstream f(fname.c_str());
    if (!f) {
        cerr << "ERROR: " << fname << " could not be opened." << endl;
        cerr << "Exiting now." << endl;
        exit(1);
    }
    p.loadTsv(f);
    f.close();
}

int extractIncrements(int i) {
    int increment = 0;
    while (tIncrements.size() != 0 && tIncrements.front() == i) {
        increment += 1;
        tIncrements.pop_front();
    }
    return increment;
}

int main(int argc, char* argv[]) {
    parseArgs(argc, argv);
    srand(seed);
    printConfig();

    RbmPredictData predictData;
    safeLoad(predictData, testFilename);
    cout << "Done loading test data" << flush;

    RbmData data;
    safeLoad(data, trainFilename);
    cout << "\rDone loading data.     " << endl;

    Rbm* r = NULL;
    if (parallel) 
        r = new RbmParallel(nThreads, data, nHidden);
    else
        r = new Rbm(data, nHidden);
    r->momentum = initialMomentum;
    r->hBiasLearnRate = hLearn;
    r->vBiasLearnRate = vLearn;
    r->WlearnRate = wLearn;
    r->weightDecay = wCost;

    for (int i = 1; i <= nEpochs; i++) {
        int increment = extractIncrements(i);
        if (increment) {
            r->T += increment;
            cout << "\tT = " << r->T << endl;
        }
        if (i == finalMomentumStart)
            r->momentum = finalMomentum;

        r->performEpoch(data); 

        if (!epochsToSaveAt.empty() && epochsToSaveAt.front() == i) {
            epochsToSaveAt.pop_front();
            stringstream t;
            t << savePrefix << i;
            double d = r->predict(data, predictData, t.str());
            cout << i << ": saving to " << t.str() << "(" << d << ")" << endl;
        } else if (predictAlways) {
            double d = r->predict(data, predictData);
            cout << i << ": " << d << endl;
        } else {
            cout << i << ": omitting prediction" << endl;
        }
    }

    return 0;
}
