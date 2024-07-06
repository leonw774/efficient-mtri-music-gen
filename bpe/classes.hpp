#ifndef CLASSES_H
#define CLASSES_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <string>
#include <cstring>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <chrono>

#define SHPAE_COUNTER_TREE_BASED // Tree or Hash

// these setting must correspond to what is defined in tokens.py
#define BEGIN_TOKEN_STR         "BOS"
#define END_TOKEN_STR           "EOS"
#define SEP_TOKEN_STR           "SEP"
#define TRACK_EVENTS_CHAR       'R'
#define MEASURE_EVENTS_CHAR     'M'
#define POSITION_EVENTS_CHAR    'P'
#define TEMPO_EVENTS_CHAR       'T'
#define NOTE_EVENTS_CHAR        'N'
#define MULTI_NOTE_EVENTS_CHAR  'U'

// for convenience
#define CONT_NOTE_EVENTS_STR    "N~"

struct RelNote {
    uint8_t isCont;   // Lowest byte: Boolean
    uint8_t relDur;   // Lower middle byte
    int8_t relPitch;  // Higher middle byte
    uint8_t relOnset; // Highest byte
    // Members are ordered such that it's value is:
    //      (MSB) relOnset relPitch relDur isCont (LSB)
    // when viewed as unsigned 32bit int

    static const uint8_t onsetLimit = 0xff;
    static const uint8_t durLimit = 0xff;

    RelNote();

    RelNote(uint8_t o, uint8_t p, uint8_t d, uint8_t c);

    bool operator<(const RelNote& rhs) const;

    bool operator==(const RelNote& rhs) const;
};

typedef std::vector<RelNote> Contour;

int b36strtoi(const char* s);

std::string itob36str(int x);

std::string contour2str(const Contour& s);

unsigned int getMaxRelOffset(const Contour& s);

std::vector<Contour> getDefaultContourDict();

std::vector<Contour> readContourVocabFile(std::ifstream& inContourFile);

void writeContourVocabFile(
    std::ostream& vocabOutfile,
    const std::vector<Contour>& contourDict
);

template <>
struct std::hash<Contour> {
    size_t operator()(const Contour& s) const;
};

#ifdef SHPAE_COUNTER_TREE_BASED
    typedef std::map<Contour, unsigned int> contour_counter_t;
#else
    typedef std::unordered_map<Contour, unsigned int> contour_counter_t;
#endif

struct MultiNote {
    uint32_t onset;
    // onsetLimit: When tpq is 12, 0x7ffffff ticks is 1,491,308 minutes in
    //             the tempo of 120 quarter note per minute,
    static const uint32_t onsetLimit = 0x7fffffff;
    // contourIndex: The index of contour in the contourDict.
    //             0: DEFAULT_REGULAR, 1: DEFAULT_CONTINUING
    //             So iterNum cannot be greater than 0xffff - 2 = 65534
    uint16_t contourIndex;
    static const uint16_t contourIndexLimit = 0xffff;
    uint8_t pitch;  // pitch shift
    uint8_t stretch;// time stretch
    uint8_t vel;

    // This `neighbor` store relative index from this multinote to others.
    // If neighbor > 0, any multinote in index (i+1) ~ (i+neighbor) 
    // is the i-th multinote's neighbor.
    uint8_t neighbor;
    static const uint8_t neighborLimit = 0x7f;

    MultiNote(bool isCont, uint32_t o, uint8_t p, uint8_t d, uint8_t v);

    bool operator<(const MultiNote& rhs) const;

    bool operator==(const MultiNote& rhs) const;
};

typedef std::vector<MultiNote> Track;

void printTrack(
    const Track& track,
    const std::vector<Contour>& contourDict,
    const size_t begin,
    const size_t length
);


struct TimeStructToken {
    uint32_t onset;

    // MSB ---------> LSB
    // T DDD NNNNNNNNNNNN
    // When T is 1, this is a tempo token, the tempo value is N. D is 0
    // When T is 0, this is a measure token,
    // the denominator is 2 to the power of D and the numerator is N
    uint16_t data;

    TimeStructToken(uint32_t o, bool t, uint16_t n, uint16_t d);
    bool isTempo() const;
    int getD() const;
    int getN() const;
};

struct Corpus {
    // Multi-notes
    std::vector<Track> *mns;
    // Time structures
    std::vector<TimeStructToken> *timeStructLists;
    // Track-program mappings
    std::vector<uint8_t> *trackInstrMaps;
    unsigned int pieceNum;

    Corpus(unsigned int pieceNum);
    ~Corpus();

    void shrink(unsigned int i);
    size_t getMultiNoteCount();
    void sortAllTracks();
};

std::map<std::string, std::string> readParasFile(std::ifstream& paraFile);

Corpus readCorpusFile(std::ifstream& corpusFile, int tpq, int maxTrackNum);

void writeOutputCorpusFile(
    std::ostream& tokenizedCorpusFile,
    const Corpus& corpus,
    const std::vector<Contour>& contourDict,
    int maxTrackNum
);

#endif
