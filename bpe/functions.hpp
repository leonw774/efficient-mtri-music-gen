#ifndef FUNCS_H
#define FUNCS_H

// return sum of all note's neighbor number
size_t updateNeighbor(
    Corpus& corpus,
    const std::vector<Contour>& contourDict,
    unsigned int gapLimit
);

Contour getContourOfMultiNotePair(
    const MultiNote& lmn,
    const MultiNote& rmn,
    const std::vector<Contour>& contourDict
);

// ignoreVelcocity is default false
double calculateAvgMulpiSize(const Corpus& corpus, bool ignoreVelocity=false);

double calculateMultinoteEntropy(const Corpus& corpus, size_t multinoteCount);

typedef std::vector<std::pair<Contour, unsigned int>> flat_contour_counter_t;

flat_contour_counter_t getContourCounter(
    Corpus& corpus,
    const std::vector<Contour>& contourDict,
    const std::string& adjacency,
    const double samplingRate
);

std::pair<Contour, unsigned int> findMaxValPair(
    const flat_contour_counter_t& contourCounter
);

#endif
