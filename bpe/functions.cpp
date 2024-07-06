#include "classes.hpp"
#include "functions.hpp"

#include <random>
#include "omp.h"

/*
    binary gcd
    https://hbfs.wordpress.com/2013/12/10/the-speed-of-gcd/
    use gcc build-in function __builtin_ctz
*/
unsigned int gcd(unsigned int a, unsigned int b) {
    if (a == 0) return b;
    if (b == 0) return a;
    unsigned int shift = __builtin_ctz(a|b);
    a >>= shift;
    do {
        b >>= __builtin_ctz(b);
        if (a > b) {
            std::swap(a, b);
        }
        b -= a;
    } while (b);
    return a << shift;
}

unsigned int gcd(unsigned int* arr, unsigned int size) {
    int g = arr[0];
    for (int i = 1; i < size; ++i) {
        if (arr[i] != 0) {
            g = gcd(g, arr[i]);
            if (g == 1) break;
        }
    }
    return g;
}

/*  
    Do merging in O(logt) time, t is arraySize
    A merge between two counter with size of A and B takes
    $\sum_{i=A}^{A+B} \log{i}$ time
    Since $\int_A^{A+B} \log{x} dx = (A+B)\log{A+B} - A\log{A} - B$
    We could say a merge is O(n logn), where n is number of elements to count
    so that the total time complexity is O(n logn logt)

    This function alters the input array.
    The return index is the index of the counter merged all other counters.
*/
int mergeCounters(contour_counter_t counterArray[], size_t arraySize) {
    std::vector<int> mergingMapIndices(arraySize);
    for (int i = 0; i < arraySize; ++i) {
        mergingMapIndices[i] = i;
    }
    while (mergingMapIndices.size() > 1) {
        // count from back to not disturb the index number when erasing
        #pragma omp parallel for
        for (int i = mergingMapIndices.size() - 1; i > 0; i -= 2) {
            int a = mergingMapIndices[i];
            int b = mergingMapIndices[i-1];
            for (
                auto it = counterArray[a].cbegin();
                it != counterArray[a].cend();
                it++
            ) {
                counterArray[b][it->first] += it->second;
            }
        }
        for (int i = mergingMapIndices.size() - 1; i > 0; i -= 2) {
            mergingMapIndices.erase(mergingMapIndices.begin()+i);
        }
    }
    return mergingMapIndices[0];
}


size_t updateNeighbor(
    Corpus& corpus,
    const std::vector<Contour>& contourDict,
    unsigned int gapLimit
) {
    size_t totalNeighborNumber = 0;
    // calculate the relative offset of all contours in contourDict
    std::vector<unsigned int> relOffsets(contourDict.size(), 0);
    for (int t = 0; t < contourDict.size(); ++t) {
        relOffsets[t] = getMaxRelOffset(contourDict[t]);
    }
    // for each piece
    #pragma omp parallel for reduction(+: totalNeighborNumber)
    for (int i = 0; i < corpus.pieceNum; ++i) {
        // for each track
        for (Track& track: corpus.mns[i]) {
            // for each multinote
            for (int k = 0; k < track.size(); ++k) {
                // printTrack(corpus.piecesMN[i][j], contourDict, k, 1);
                unsigned int onsetTime = track[k].onset;
                unsigned int offsetTime = onsetTime + (
                    relOffsets[track[k].contourIndex] * track[k].stretch
                );
                unsigned int immdFollowOnset = 0;
                int n = 1;
                while (k+n < track.size() && n < MultiNote::neighborLimit) {
                    unsigned int nOnsetTime = track[k+n].onset;
                    // immediately following
                    if (nOnsetTime >= offsetTime) { 
                        if (immdFollowOnset == 0) {
                            if (nOnsetTime - offsetTime > gapLimit) {
                                break;
                            }
                            immdFollowOnset = nOnsetTime;
                        }
                        else if (nOnsetTime != immdFollowOnset) {
                            break;
                        }
                    }
                    // overlapping
                    // else {
                    //     /* do nothing */
                    // }
                    n++;
                }
                track[k].neighbor = n - 1;
                totalNeighborNumber += n - 1;
            }
        }
    }
    return totalNeighborNumber;
}

/* Return empty contour if cannot find valid contour */
Contour getContourOfMultiNotePair(
    const MultiNote& lmn,
    const MultiNote& rmn,
    const std::vector<Contour>& contourDict
) {
    if (rmn.onset < lmn.onset) {
        // return getContourOfMultiNotePair(rmn, lmn, contourDict);
        throw std::runtime_error(
            "right multi-note has smaller onset than left multi-note");
    }
    const Contour lContour = contourDict[lmn.contourIndex],
                rContour = contourDict[rmn.contourIndex];
    const int rSize = rContour.size(), lSize = lContour.size();
    const unsigned int rightStretch = rmn.stretch;
    const int32_t deltaOnset = (int32_t) rmn.onset - (int32_t) lmn.onset;

    // times = {
    //   right_onset_1, ..., right_onset_n,
    //   right_stretch_1, ..., right_stretch_n,
    //   left_stretch
    // }
    unsigned int times[rSize*2+1]; 
    int t = 0;
    for (const RelNote& rRelNote: rContour) {
        times[t] = rRelNote.relOnset * rightStretch + deltaOnset;
        times[t+rSize] = (unsigned int) rRelNote.relDur * rightStretch;
        t++;
    }
    times[rSize*2] = lmn.stretch;
    unsigned int newStretch = gcd(times, rSize*2+1);

    // re-calculate the new time values
    for (unsigned int& time: times) {
        time /= newStretch;
    }
    const unsigned int lStretchRatio = times[rSize*2];

    // check to prevent overflow
    // if overflowed, return empty contour
    for (int i = 0; i < rSize; ++i) {
        if (times[i] > RelNote::onsetLimit) {
            return Contour();
        }
    }
    unsigned int newLRelOnsets[lSize];
    unsigned int newLRelDur[lSize];
    t = 0;
    for (const RelNote& lRelNote: lContour) {
        newLRelOnsets[t] = lRelNote.relOnset * lStretchRatio;
        newLRelDur[t] = lRelNote.relDur * lStretchRatio;
        if (newLRelOnsets[t] > RelNote::onsetLimit
            || newLRelDur[t] > RelNote::durLimit) {
            return Contour();
        }
        t++;
    }

    const int8_t deltaPitch = (int8_t) rmn.pitch - (int8_t) lmn.pitch;
    // like the merge part of merge sort
    int lpos = 0, rpos = 0;
    RelNote newLRel(
        newLRelOnsets[0],
        lContour[0].relPitch,
        newLRelDur[0],
        lContour[0].isCont
    );
    RelNote newRRel(
        times[0],
        rContour[0].relPitch + deltaPitch,
        times[rSize],
        rContour[0].isCont
    );
    Contour pairContour(lSize + rSize);
    enum MergeCase {LEFT, RIGHT};
    while (lpos < lSize || rpos < rSize) {
        int mergeCase = 0;
        if (rpos == rSize) {
            mergeCase = MergeCase::LEFT;
        }
        else if (lpos == lSize) {
            mergeCase = MergeCase::RIGHT;
        }
        else {
            if (newLRel < newRRel) {
                mergeCase = MergeCase::LEFT;
            }
            else {
                mergeCase = MergeCase::RIGHT;
            }
        }

        if (mergeCase == MergeCase::LEFT) {
            pairContour[lpos+rpos] = newLRel;
            lpos++;
            if (lpos != lSize) {
                newLRel = RelNote(
                    newLRelOnsets[lpos],
                    lContour[lpos].relPitch,
                    newLRelDur[lpos],
                    lContour[lpos].isCont
                );
            }
        }
        else {
            pairContour[lpos+rpos] = newRRel;
            rpos++;
            if (rpos != rSize) {
                newRRel = RelNote(
                    times[rpos],
                    rContour[rpos].relPitch + deltaPitch,
                    times[rpos+rSize],
                    rContour[rpos].isCont
                );
            }
        }
    }
    return pairContour;
}

// ignoreVelcocity is default false
double calculateAvgMulpiSize(const Corpus& corpus, bool ignoreVelcocity) {
    unsigned int max_thread_num = omp_get_max_threads();
    std::vector<uint8_t> multipiSizes[max_thread_num];
    #pragma omp parallel for
    for (int i = 0; i < corpus.pieceNum; ++i) {
        int thread_num = omp_get_thread_num();
        std::vector<Track>& piece = corpus.mns[i];
        // for each track
        for (const Track &track: piece) {
            // key: 64 bits
            //     16 unused, 8 for velocity, 8 for time stretch, 32 for onset
            // value: occurence count
            std::map<uint64_t, uint8_t> curTrackMulpiSizes;
            for (int k = 0; k < track.size(); ++k) {
                uint64_t key = track[k].onset;
                key |= ((uint64_t) track[k].stretch) << 32;
                if (!ignoreVelcocity) {
                    key |= ((uint64_t) track[k].vel) << 40;
                }
                curTrackMulpiSizes[key] += 1;
            }
            for (
                auto it = curTrackMulpiSizes.cbegin();
                it != curTrackMulpiSizes.cend();
                ++it
            ) {
                multipiSizes[thread_num].push_back(it->second);
            }
        }
    }
    size_t accumulatedMulpiSize = 0;
    size_t mulpiCount = 0;
    for (int i = 0; i < max_thread_num; ++i) {
        for (int j = 0; j < multipiSizes[i].size(); ++j) {
            accumulatedMulpiSize += multipiSizes[i][j];
        }
        mulpiCount += multipiSizes[i].size();
    }
    return (double) accumulatedMulpiSize / (double) mulpiCount;
}

/*
We assume the entropy of a collection of multi-note tokens to be the entropy
of the collection of all its members' attribute values. That is:

    - \sum_{x \in (contours \cup pitches \cup streches \cup velocities)} {
        p(x) * log(p(x))
    }
*/
double calculateMultinoteEntropy(const Corpus& corpus, size_t multinoteCount) {
    // The four maps are for {contour, pitch, stretch, velocity}
    std::map<uint16_t, size_t> tokenDistributions[4];
    #pragma omp parallel for
    for (int a = 0; a < 4; a++) {
        for (int i = 0; i < corpus.pieceNum; ++i) {
            std::vector<Track>& piece = corpus.mns[i];
            // for each track
            for (const Track &track: piece) {
                for (int k = 0; k < track.size(); ++k) {
                    uint16_t keys[4] = {
                        track[k].contourIndex,
                        track[k].pitch,
                        track[k].stretch,
                        track[k].vel
                    };
                    tokenDistributions[a][keys[a]] += 1;
                }
            }
        }
    }
    double entropy = 0;
    double logSize = log2(4 * multinoteCount);
    for (int a = 0; a < 4; a++) {
        for (const auto& kv: tokenDistributions[a]) {
            double p = kv.second;
            entropy += p * (log2(p) - logSize);
        }
    }
    entropy /= 4 * multinoteCount;
    return -entropy;
}

// Fisher-Yates shuffle
// https://stackoverflow.com/questions/9345087
template<class BidiIter>
BidiIter random_unique(BidiIter begin, BidiIter end, size_t num_random) {
    size_t left = std::distance(begin, end);
    while (num_random--) {
        BidiIter r = begin;
        std::advance(r, rand() % left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

flat_contour_counter_t getContourCounter(
    Corpus& corpus,
    const std::vector<Contour>& contourDict,
    const std::string& adjacency,
    const double samplingRate
) {
    if (samplingRate <= 0 || 1 < samplingRate) {
        throw std::runtime_error(
            "samplingRate in contourScoring not in range (0, 1]");
    }
    bool isOursAdj = (adjacency == "ours");

    std::set<std::pair<int, int>> sampledTracks;
    if (samplingRate < 1.0) {
        std::vector<std::pair<int, int>> allTracks;
        for (int i = 0; i < corpus.pieceNum; ++i) {
            for (int j = 0; j < corpus.mns[i].size(); ++j) {
                allTracks.push_back({i, j});
            }       
        }
        size_t sampleNum = allTracks.size() * samplingRate + 0.5;
        if (sampleNum == 0) {
            sampleNum = 1;
        }
        random_unique(allTracks.begin(), allTracks.end(), sampleNum);
        sampledTracks = std::set<std::pair<int, int>>(
            allTracks.begin(), allTracks.begin() + sampleNum
        );
    }

    unsigned int max_thread_num = omp_get_max_threads();
    contour_counter_t contourCountersParallel[max_thread_num];
    #pragma omp parallel for
    for (int i = 0; i < corpus.pieceNum; ++i) {
        int thread_num = omp_get_thread_num();
        contour_counter_t& myContourCounter =
            contourCountersParallel[thread_num];
        std::vector<Track>& piece = corpus.mns[i];
        // for each track
        for (int j = 0; j < piece.size(); ++j) {
            if (samplingRate < 1.0 && sampledTracks.count({i, j}) == 0) {
                continue;
            }
            Track& track = piece[j];
            /* NO DOING THIS BECAUSE VERY SLOW */
            // std::map<Contour, std::unordered_set<int>> contour2Indices;
            // for each multinote
            for (int k = 0; k < track.size(); ++k) {
                // for each neighbor
                const MultiNote& curMN = track[k];
                for (int n = 1; n <= curMN.neighbor; ++n) {
                    const MultiNote& neighborMN = track[k+n];
                    if (isOursAdj) {
                        if (curMN.vel != neighborMN.vel) continue;
                    }
                    else {
                        // mulpi
                        if (curMN.onset != neighborMN.onset) break;
                        if (curMN.vel != neighborMN.vel) continue;
                        if (curMN.stretch != neighborMN.stretch) continue;
                    }
                    Contour contour = getContourOfMultiNotePair(
                        curMN,
                        neighborMN,
                        contourDict
                    );
                    // empty contour mean overflow happened
                    if (contour.size() == 0) continue;

                    /* NO DOING THIS BECAUSE VERY SLOW */
                    // // fix issue of invalid count from used multi-notes
                    // std::unordered_set<int>& idxs = contour2Indices[contour];
                    // // if any source already used by this contour, ignore
                    // if (idxs.count(k) || idxs.count(k+n)) continue;
                    // // else insert the sources into used indices
                    // idxs.insert({k, k+n});

                    myContourCounter[contour] += 1;
                }
            }
        }
    }

    int mergedIndex = mergeCounters(contourCountersParallel, max_thread_num);
    flat_contour_counter_t contourCounter;
    contourCounter.reserve(contourCountersParallel[mergedIndex].size());
    contourCounter.assign(
        contourCountersParallel[mergedIndex].cbegin(),
        contourCountersParallel[mergedIndex].cend()
    );
    return contourCounter;
}


std::pair<Contour, unsigned int> findMaxValPair(
    const flat_contour_counter_t& contourCounter
) {
    #pragma omp declare reduction\
        (maxsecond: std::pair<Contour, unsigned int>:\
            omp_out = omp_in.second > omp_out.second ? omp_in : omp_out\
        )
    std::pair<Contour, unsigned int> maxSecondPair;
    maxSecondPair.second = 0;
    #pragma omp parallel for reduction(maxsecond: maxSecondPair)
    for (int i = 0; i < contourCounter.size(); ++i) {
        if (contourCounter[i].second > maxSecondPair.second) {
            maxSecondPair = contourCounter[i];
        }
    }
    return maxSecondPair;
}
