#include "rbmdata.h"
#include <vector>
#include <string>

RbmData::RbmData() {

}

RbmData::~RbmData() {

}

void RbmData::loadTsv(istream& data) {
    vector<int> range(1, 0); // range of first user starts at 0
    vector<int> ratings;
    vector<int> movies;

    int previousUser = 0;
    float maxRating = 0.0;
    int maxMovie = 0;
    int moviesRatedByUser = 0;
    int user, movie;
    float rating;
    while (true) {
        data >> user >> movie >> rating; 
        if (user != previousUser || data.eof()) {
            range.push_back(range[previousUser] + moviesRatedByUser);
            previousUser = user;
            moviesRatedByUser = 0;
        }
        if (data.eof())
            break;
        maxRating = max(maxRating, rating);
        ratings.push_back(rating);
        maxMovie = max(maxMovie, movie);
        movies.push_back(movie);

        moviesRatedByUser += 1;
    }

    nClasses = maxRating;
    this->range.resize(range.size());
    for (size_t i = 0; i < range.size(); i++)
        this->range(i) = range[i] * nClasses;

    this->ratings.setZero(ratings.size() * nClasses);
    for (size_t i = 0; i < ratings.size(); i++)
        this->ratings(i * nClasses + ratings[i] - 1) = 1;

    nMovies = maxMovie + 1;
    this->movies.setZero(movies.size() * nClasses);
    int m = 0;
    for (size_t i = 0; i < movies.size(); i++) {
        for (int c = 0; c < nClasses; c++) {
            this->movies(m) = movies[i] * nClasses + c;
            m++;
        }
    }
}

ostream& operator<<(ostream& out, const RbmData& d) {
    out << "range = " << endl;
    out << d.range << endl;
    out << "ratings = " << endl;
    out << d.ratings << endl;
    out << "movies = " << endl;
    out << d.movies; 

    return out;
}

