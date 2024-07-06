#include "classes.hpp"

/********
  RelNote
********/

RelNote::RelNote() : relOnset(0), relPitch(0), relDur(0), isCont(0) {};

RelNote::RelNote(uint8_t o, uint8_t p, uint8_t d, uint8_t c) {
    relOnset = o;
    relPitch = p;
    relDur = d;
    isCont = c;
}

// Compare on relOnset first and then relPitch, relDur, isCont.
// This decide how contour set should be represented sequentially
// mainly when output to contour_vocab and vocab.json
// This order allows faster computation because the zero-point
// would be the first element
bool RelNote::operator<(const RelNote& rhs) const {
    // if (relOnset != rhs.relOnset) return relOnset < rhs.relOnset;
    // if (relPitch != rhs.relPitch) return relPitch < rhs.relPitch;
    // if (relDur != rhs.relDur) return relDur < rhs.relDur;
    // return isCont < rhs.isCont;

    // wicked casting: RelNote -> uint32_t
    return *((uint32_t*) this) < *((uint32_t*) &(rhs));
}

bool RelNote::operator==(const RelNote& rhs) const {
    // return (relOnset == rhs.relOnset
    //     && relPitch == rhs.relPitch
    //     && relDur == rhs.relDur
    //     && isCont == rhs.isCont);
    return *((uint32_t*) this) == *((uint32_t*) &(rhs));
}

/********
  Contour
********/

int b36strtoi(const char* s) {
    unsigned int l = strlen(s);
    bool isNeg = s[0] == '-';
    int r = 0;
    for (unsigned int i = (isNeg ? 1 : 0); i < l; ++i) {
        if ('0' <= s[i] && s[i] <= '9') {
            r = (r * 36) + s[i] - '0';
        }
        else if ('A' <= s[i] && s[i] <= 'Z') {
            r = (r * 36) + s[i] - 'A' + 10;
        }
    }
    return (isNeg ? -r : r);
}

int b36strtoi(const std::string& s) {
    return b36strtoi(s.c_str());
}

std::string itob36str(int x) {
    if (0 <= x && x < 36) {
        return std::string(1, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[x]);
    }
    bool isNeg = x < 0;
    if (isNeg) {
        x = -x;
    }
    std::string r;
    while (x) {
        r = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[x%36] + r;
        x /= 36;
    }
    return (isNeg ? "-" : "") + r;
}

std::string contour2str(const Contour& s) {
    std::stringstream ss;
    for (const RelNote& r: s) {
        ss << itob36str(r.relOnset)
           << "," << itob36str(r.relPitch)
           << "," << itob36str(r.relDur) << (r.isCont ? "~" : "") << ";";
    }
    return ss.str();
}

unsigned int getMaxRelOffset(const Contour& s) {
    unsigned int maxRelOffset = 0;
    for (const RelNote& r: s) {
        if (maxRelOffset < r.relOnset + (unsigned int) r.relDur) {
            maxRelOffset = r.relOnset + (unsigned int) r.relDur;
        }
    }
    return maxRelOffset;
}

std::vector<Contour> getDefaultContourDict() {
    return {
        {RelNote(0, 0, 1, 0)}, // DEFAULT_REGULAR
        {RelNote(0, 0, 1, 1)}  // DEFAULT_CONTINUING
    };
}

std::vector<Contour> readContourVocabFile(std::ifstream& inContourFile) {
    std::vector<Contour> contourDict = getDefaultContourDict();
    inContourFile.seekg(0, std::ios::beg);
    std::string line;
    while (inContourFile.good()) {
        std::getline(inContourFile, line, '\n');
        if (line.size() == 0) continue;
        line.pop_back(); // because last character must be ';'
        for (int i = 0; i < line.size(); ++i) {
            if (line[i] == ',' || line[i] == ';') {
                line[i] = ' ';
            }
        }
        Contour contour;
        std::stringstream lineSS(line);
        while(lineSS.good()) {
            int isCont = 0, relOnset, relPitch, relDur;
            std::string relOnsetStr, relPitchStr, relDurStr;
            lineSS >> relOnsetStr >> relPitchStr >> relDurStr;
            if (relDurStr.back() == '~') {
                isCont = 1;
                relDurStr.pop_back();
            }
            relOnset = b36strtoi(relOnsetStr.c_str());
            relPitch = b36strtoi(relPitchStr.c_str());
            relDur = b36strtoi(relDurStr.c_str());
            contour.push_back(RelNote(relOnset, relPitch, relDur, isCont));
        }
        // std::cout << contour2str(contour) << '\n' << line << std::endl;
        contourDict.push_back(contour);
        line.clear();
    }
    return contourDict;
}

void writeContourVocabFile(
    std::ostream& vocabOutfile,
    const std::vector<Contour>& contourDict
) {
    // dont need to write the first 2 default contour
    for (int i = 2; i < contourDict.size(); ++i) {
        std::stringstream ss;
        ss << contour2str(contourDict[i]);
        vocabOutfile << ss.str() << std::endl;
    }
}

size_t std::hash<Contour>::operator()(const Contour& s) const {
    size_t h = 0;
    uint32_t x;
    for (int i = 0; i < s.size(); ++i) {
        x = *((uint32_t*) &s[i]);
        // formula from boost::hash_combine
        // take away the `hash()` to reduce compute time
        // h ^= hash<uint32_t>()(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= x + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
}


/********
  MultiNote
********/

MultiNote::MultiNote(bool c, uint32_t o, uint8_t p, uint8_t d, uint8_t v) {
    if (o > onsetLimit) {
        throw std::runtime_error("MultiNote onset exceed limit.");
    }
    contourIndex = c ? 1 : 0;
    onset = o;
    pitch = p;
    stretch = d;
    vel = v;
    neighbor = 0;
}

bool MultiNote::operator<(const MultiNote& rhs) const {
    if (onset != rhs.onset) return onset < rhs.onset;
    return pitch < rhs.pitch;
}

bool MultiNote::operator==(const MultiNote& rhs) const {
    return contourIndex == rhs.contourIndex 
        && onset == rhs.onset
        && pitch == rhs.pitch
        && stretch == rhs.stretch
        && vel == rhs.vel;
}

void printTrack(
    const Track& track,
    const std::vector<Contour>& contourDict,
    const size_t begin,
    const size_t length
) {
    for (int i = begin; i < begin + length; ++i) {
        std::cout << i << " -"
            << " Contour=" << contour2str(contourDict[track[i].contourIndex])
            << " onset=" << (int) track[i].onset
            << " pitch=" << (int) track[i].pitch
            << " stretch=" << (int) track[i].stretch
            << " velocity=" << (int) track[i].vel
            << " neighbor=" << (int) track[i].neighbor
            << std::endl;
    }
}

/********
  TimeStructToken
********/

TimeStructToken::TimeStructToken(uint32_t o, bool t, uint16_t n, uint16_t d) {
    onset = o;
    if (t) {
        data = n;
        data |= 1 << 15; 
    }
    else {
        data = n;
        switch (d) {
            case 2:  data |= (1 << 12); break;
            case 4:  data |= (2 << 12); break;
            case 8:  data |= (3 << 12); break;
            case 16: data |= (4 << 12); break;
            case 32: data |= (5 << 12); break;
            case 64: data |= (6 << 12); break;
            case 128: data |= (7 << 12); break;
        }
    }
}

bool TimeStructToken::isTempo() const {
    return data >> 15;
}

// dont "getD" if you are tempo token
int TimeStructToken::getD() const { 
    return 1 << (data >> 12);
}

int TimeStructToken::getN() const { 
    return data & 0x0fff;
}

/********
  Corpus
********/

Corpus::Corpus(unsigned int _size) {
    pieceNum = _size;
    mns = new std::vector<Track>[_size];
    timeStructLists = new std::vector<TimeStructToken>[_size];
    trackInstrMaps = new std::vector<uint8_t>[_size];
}

Corpus::~Corpus() {
    delete[] mns;
    delete[] timeStructLists;
    delete[] trackInstrMaps;
}

void Corpus::shrink(unsigned int i) {
    for (Track& track: mns[i]) {
        track.shrink_to_fit();
    }
    mns[i].shrink_to_fit();
    timeStructLists[i].shrink_to_fit();
}

size_t Corpus::getMultiNoteCount() {
    size_t multinoteCount = 0;
    for (int i = 0; i < pieceNum; ++i) {
        std::vector<Track>& piece = mns[i];
        for (int j = 0; j < piece.size(); ++j) {
            multinoteCount += piece[j].size();
        }
    }
    return multinoteCount;
}

void Corpus::sortAllTracks() {
    for (int i = 0; i < pieceNum; ++i) {
        std::vector<Track>& piece = mns[i];
        for (int j = 0; j < piece.size(); ++j) {
            std::sort(piece[j].begin(), piece[j].end());
        }
    }
}

/********
 Corpus's IO
********/

std::map<std::string, std::string> readParasFile(std::ifstream& parasFile) {
    if (!parasFile) {
        throw std::runtime_error("Could not open file");
    }
    parasFile.seekg(0, std::ios::beg);
    std::string line, key, value;
    std::stringstream ss;
    std::map<std::string, std::string> resultMap;
    while (parasFile.good()) {
        std::getline(parasFile, line, '\n');
        // std::cout << "line = " << line << std::endl;
        ss << line;
        ss >> key;
        // std::cout << "key = " << key << std::endl;
        if (ss >> value) {
            // std::cout << "value = " << value << std::endl;
            if (key != "-"){
                key.pop_back(); // remove last character because it is ':'
                resultMap[key] = value;
            }
        }
        ss.clear();
    }
    return resultMap;
}

Corpus readCorpusFile(std::ifstream& inCorpusFile, int tpq, int maxTrackNum) {
    // reset flags
    inCorpusFile.clear();
    inCorpusFile.seekg(0, std::ios::beg);

    // get file size
    // inCorpusFile.seekg(0, std::ios::end);
    // std::streampos filesize = inCorpusFile.tellg();
    // inCorpusFile.seekg(0, std::ios::beg);

    // count lines and spaces
    std::vector<int> tokenNums;
    while (inCorpusFile.good()) {
        std::string line;
        getline(inCorpusFile, line);
        if (line.size() == 0) {
            break;
        }
        tokenNums.push_back(std::count(line.begin(), line.end(), ' '));
    }
    Corpus corpus(tokenNums.size());
    // std::cout << corpus.pieceNum << std::endl;

    // reset flags again
    inCorpusFile.clear();
    inCorpusFile.seekg(0, std::ios::beg);

    unsigned int curPieceIdx = 0;
    uint32_t curMeasureStart = 0, curMeasureLength = 0, curTime = 0;
    while (inCorpusFile.good()) {
        unsigned char c = inCorpusFile.get(), i;
        char a[8];
        switch (c) {
            case BEGIN_TOKEN_STR[0]: // BOS
                while (inCorpusFile.get() != ' ');
                corpus.mns[curPieceIdx].reserve(maxTrackNum);
                corpus.trackInstrMaps[curPieceIdx].reserve(maxTrackNum);
                curMeasureStart = curMeasureLength = curTime = 0;
                break;

            case END_TOKEN_STR[0]: // EOS
                corpus.shrink(curPieceIdx);
                curPieceIdx++;
                while (inCorpusFile.get() != '\n');
                break;

            case SEP_TOKEN_STR[0]: // SEP
                while (inCorpusFile.get() != ' ');
                for (Track &t: corpus.mns[curPieceIdx]) {
                    t.reserve(tokenNums[curPieceIdx]);
                }
                corpus.timeStructLists[curPieceIdx].reserve(
                    tokenNums[curPieceIdx]
                );
                break;

            case TRACK_EVENTS_CHAR:
                corpus.mns[curPieceIdx].push_back(Track());
                inCorpusFile.getline(a, 8, ':');
                corpus.trackInstrMaps[curPieceIdx].push_back(
                    (uint8_t) b36strtoi(a)
                );
                while (inCorpusFile.get() != ' '); // eat the track number
                break;

            case MEASURE_EVENTS_CHAR:
                curMeasureStart += curMeasureLength;
                uint8_t numer, denom;
                inCorpusFile.getline(a, 8, '/');
                numer = b36strtoi(a);
                inCorpusFile.getline(a, 8, ' ');
                denom = b36strtoi(a);
                curMeasureLength = 4 * tpq * numer / denom;
                corpus.timeStructLists[curPieceIdx].push_back(
                    TimeStructToken(curMeasureStart, false, numer, denom)
                );
                break;

            case TEMPO_EVENTS_CHAR:
                inCorpusFile.getline(a, 8, ' ');
                corpus.timeStructLists[curPieceIdx].push_back(
                    TimeStructToken(curTime, true, b36strtoi(a), 0)
                );
                break;

            case POSITION_EVENTS_CHAR:
                inCorpusFile.getline(a, 8, ' ');
                curTime = curMeasureStart + b36strtoi(a);
                break;

            case NOTE_EVENTS_CHAR:
                uint8_t isCont, p, d, v, t;
                if (isCont = (inCorpusFile.get() == '~')) {
                    inCorpusFile.get();
                }
                inCorpusFile.getline(a, 8, ':');
                p = b36strtoi(a);
                inCorpusFile.getline(a, 8, ':');
                d = b36strtoi(a);
                inCorpusFile.getline(a, 8, ':');
                v = b36strtoi(a);
                inCorpusFile.getline(a, 8, ' ');
                t = b36strtoi(a);
                corpus.mns[curPieceIdx][t].push_back(
                    MultiNote(isCont, curTime, p, d, v)
                );
                break;
            
            case 255: // is -1, means EOF
                break;

            default:
                std::cout << "Corpus format error: Token starts with "
                    << (int) c << "\n";
                throw std::runtime_error("Corpus format error");
                
        }
    }
    return corpus;
}

void writeOutputCorpusFile(
    std::ostream& outCorpusFile,
    const Corpus& corpus,
    const std::vector<Contour>& contourDict,
    int maxTrackNum
) {
    // cursor: point to the next multi-note or time structure to use
    // end: cursor of the track or time structures points to the end

    int trackCursors[maxTrackNum]; 
    bool trackEnds[maxTrackNum];
    int trackEndsCount;

    // ts = time structures (measure and tempo)
    int tsCursor;
    bool tsEnd;

    std::vector<std::string> contourStrList;
    for (const Contour& s: contourDict) {
        contourStrList.push_back(contour2str(s));
    }


    for (int i = 0; i < corpus.pieceNum; ++i) {
        outCorpusFile << BEGIN_TOKEN_STR << " ";
        for (int j = 0; j < corpus.trackInstrMaps[i].size(); ++j) {
            outCorpusFile << TRACK_EVENTS_CHAR
                << itob36str(corpus.trackInstrMaps[i][j])
                << ":" << itob36str(j) << " ";
        }
        outCorpusFile << SEP_TOKEN_STR << " ";

        int curMeasureStart = 0; // onset time of previous measure event
        int prevPosOnset = -1; // onset time of previous position event
        unsigned int trackNum = corpus.mns[i].size();
        memset(trackCursors, 0, sizeof(trackCursors));
        memset(trackEnds, 0, sizeof(trackEnds));
        tsEnd = false;
        tsCursor = trackEndsCount = 0;

        // should there is be some tracks with no notes
        // should have eliminated them in pre-processing
        // but just keep it here for safety
        for (int j = 0; j < corpus.mns[i].size(); ++j) {
            trackEnds[j] = (corpus.mns[i][j].size() == 0);
        }

        while ((trackEndsCount < trackNum) || !tsEnd) {
            // find the multi-note token with smallest onset
            int minMNOnset = INT32_MAX;
            // record the track with earliest onset
            // use the first appearence if there is more than one
            int minOnsetTrack = 0; 
            for (int j = 0; j < trackNum; ++j) {
                if (trackEnds[j]) continue;
                if (minMNOnset > corpus.mns[i][j][trackCursors[j]].onset) {
                    minMNOnset = corpus.mns[i][j][trackCursors[j]].onset;
                    minOnsetTrack = j;
                }
            }
            // if ts's onset <= mn's onset, ts first
            const TimeStructToken& curTS = corpus.timeStructLists[i][tsCursor];
            if (!tsEnd && curTS.onset <= minMNOnset) {
                // std::cout << "TS" << i << "," << tsCursor << ",onset="
                //     << corpus.times[i][tsCursor].onset << std::endl;
                if (curTS.isTempo()) {
                    // is tempo
                    if (prevPosOnset < (int) curTS.onset) {
                        outCorpusFile << POSITION_EVENTS_CHAR
                            << itob36str(curTS.onset - curMeasureStart)
                            << " ";
                        prevPosOnset = curTS.onset;
                    }
                    outCorpusFile << TEMPO_EVENTS_CHAR
                        << itob36str(curTS.getN()) << " ";
                }
                else {
                    // is measure
                    outCorpusFile << MEASURE_EVENTS_CHAR
                        << itob36str(curTS.getN()) << "/"
                        << itob36str(curTS.getD()) << " ";
                    curMeasureStart = curTS.onset;
                }
                tsCursor++;
                if (tsCursor == corpus.timeStructLists[i].size()) {
                    tsEnd = true;
                }
            }
            else {
                const MultiNote& curMN = 
                    corpus.mns[i][minOnsetTrack][trackCursors[minOnsetTrack]];
                // std::cout << "MN " << i << "," << minOnsetTrack
                //     << "," << trackCursors[minOnsetTrack]
                //     << ", onset=" << curMN.onset << std::endl;
                if (prevPosOnset < (int) minMNOnset) {
                    outCorpusFile << POSITION_EVENTS_CHAR
                        << itob36str(minMNOnset - curMeasureStart) << " ";
                    prevPosOnset = minMNOnset;
                }
                std::string contourStr;
                unsigned int contourIndex = curMN.contourIndex;
                if (contourIndex == 0) {
                    contourStr = NOTE_EVENTS_CHAR;
                }
                else if (contourIndex == 1) {
                    contourStr = CONT_NOTE_EVENTS_STR;
                }
                else {
                    contourStr = MULTI_NOTE_EVENTS_CHAR
                        + contourStrList[contourIndex];
                }
                outCorpusFile << contourStr
                    << ":" << itob36str(curMN.pitch)
                    << ":" << itob36str(curMN.stretch)
                    << ":" << itob36str(curMN.vel)
                    << ":" << itob36str(minOnsetTrack) << " ";
                trackCursors[minOnsetTrack]++;
                if (
                    trackCursors[minOnsetTrack]
                    == corpus.mns[i][minOnsetTrack].size()
                ) {
                    trackEnds[minOnsetTrack] = true;
                }
            }

            // update the trackEndsCount
            trackEndsCount = 0;
            for (int j = 0; j < trackNum; ++j) {
                if (trackEnds[j]) {
                    trackEndsCount++;
                }
            }
        }

        outCorpusFile << END_TOKEN_STR << "\n";
    }
}
