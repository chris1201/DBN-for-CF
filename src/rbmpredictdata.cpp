#include "rbmpredictdata.h"
void RbmPredictData::loadTsv(istream& data) {
    vector<int> range(1, 0); // range of first user starts at 0
    vector<int> ratings;
    vector<int> movies;

    int previousUser = 0;
    int maxRating = 0;
    int maxMovie = 0;
    int moviesRatedByUser = 0;
    int user, movie, rating;
    while (true) {
        data >> user >> movie >> rating; 
        if (user != previousUser || data.eof()) {
            if (moviesRatedByUser > 0) {
                range.push_back(range.back() + moviesRatedByUser);
                userIds.push_back(previousUser);
                moviesRatedByUser = 0;
            }
            previousUser = user;
        }
        if (data.eof())
            break;
        ratings.push_back(rating);
        movies.push_back(movie);
        maxRating = max(maxRating, rating);
        maxMovie = max(maxMovie, movie);

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

ostream& operator<<(ostream& out, const RbmPredictData& d) {
    out << "range = " << endl;
    out << d.range << endl;
    out << "ratings = " << endl;
    out << d.ratings << endl;
    out << "movies = " << endl;
    out << d.movies << endl;
    out << "userids = " << endl;
    for (auto user = d.userIds.begin(); user != d.userIds.end(); user++)
        out << *user << endl;

    return out;
}
