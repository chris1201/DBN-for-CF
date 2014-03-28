#ifndef RBM_DATA_H
#define RBM_DATA_H

#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

class RbmData {
public:
    RbmData();
    virtual ~RbmData();

    // Load data from a TSV-formatted stream:
    // userid (uint)    movieid (uint)    rating (uint)
    //
    // Preconditions:
    //      userids and movieids are continuous and start from 0
    //      file is sorted ascending on: (userid, movieid)
    virtual void loadTsv(istream& data); 

    friend ostream& operator<<(ostream& out, const RbmData& d);

    VectorXi range;
    RowVectorXf ratings;
    VectorXi movies;

    int nClasses;
    int nMovies;
};

#endif
