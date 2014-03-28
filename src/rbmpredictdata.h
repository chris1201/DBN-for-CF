#ifndef RBM_PREDICT_DATA
#define RBM_PREDICT_DATA
#include <vector>
#include "rbmdata.h"

class RbmPredictData : public RbmData {
public:
    // expected file format:
    // <user>\t<movie>\t<rating>\n
    // Preconditions:
    //      expects file sorted ascending on (user, movie) 
    void loadTsv(istream& data);

    vector<int> userIds;

    friend ostream& operator<<(ostream& out, const RbmPredictData& d);
};

#endif
