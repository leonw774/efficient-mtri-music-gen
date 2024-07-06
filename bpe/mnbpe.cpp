#include "classes.hpp"
#include "functions.hpp"
#include <string>
#include <algorithm>
#include <getopt.h>
#include "omp.h"

int main(int argc, char *argv[]) {
    /* read and validate args */

    // optionals
    int doLog = 0;
    int workerNum = -1; // -1 means use default
    std::string inContourVocabFilePath = std::string();
    int isApply = 0;
    std::string adjacency = std::string("ours");
    double samplingRate = 1.0;
    double minScore = 0.0;
    // positional (required)
    std::string inCorpusDirPath;
    std::string outCorpusDirPath;
    int iterNum = MultiNote::contourIndexLimit - 2;

    int cmd_opt = 0;
    int opt_index = 0;
    int nonopt_begin_index = 1;
    std::string cmdLineUsage = "Usage:\n"
        "mnbpe [--log] [--worker-number <number>] "
        "[--apply <contour-vocab-path>] [--adj {\"ours\"|\"mulpi\"}] "
        "[--sampling-rate <rate>] [--min-score <score>] "
        "<in-corpus-dir-path> <out-corpus-dir-path> [<iteration-number>]\n\n"
        "--log          \t Flag set to print log each iteration\n"
        "--worker-number\t Number of thread for OMP\n"
        "--adj          \t Which adjacency: \"ours\" or \"mulpi\"\n"
        "--sampling-rate\t Track sampling rate\n"
        "--min-score    \t Minimum score to early stop\n";

    const option long_options[] = {
        {"log", no_argument, &doLog, 1},
        {"worker-number", required_argument, NULL, 'w'},
        {"apply", required_argument, NULL, 'a'},
        {"adj", required_argument, NULL, 'n'},
        {"sampling-rate", required_argument, NULL, 'r'},
        {"min-score", required_argument, NULL, 'm'},
        {NULL, 0, NULL, 0}
    };

    while (1) {
        int c = getopt_long(argc, argv, "lw:a:n:r:m:",
            long_options, &opt_index);
        if (c == -1) {
            break;
        }
        switch (c) {
            case 0: // this is flag option
                break;
            case 'w':
                workerNum = atoi(optarg);
                if (workerNum <= 0) {
                    std::cout << "Error: worker-nubmer is not positive: "
                        << workerNum << std::endl;
                }
                omp_set_num_threads(workerNum);
                break;
            case 'a':
                inContourVocabFilePath = std::string(optarg);
                isApply = 1;
                break;
            case 'n':
                adjacency = std::string(optarg);
                if (adjacency != "ours" && adjacency != "mulpi") {
                    std::cout
                        << "Error: adjacency is not \"ours\" or \"mulpi\": "
                        << adjacency << std::endl;
                    return 1;
                }
                break;
            case 'r':
                samplingRate = atof(optarg);
                break;
            case 'm':
                minScore = atof(optarg);
                break;
            case '?':
                std::cout << "Bad option: "
                    << (isprint(optopt) ? (char)optopt : (int)optopt) << "\n"
                    << cmdLineUsage << std::endl;
                return 1;
            default:
                abort();
        }
    }
    if (argc - optind != 3 && argc - optind != 2) {
        std::cout << "Expect 3 or 2 non-optional arguments. "
            << "Get " << argc - optind << "\n"
            << cmdLineUsage << std::endl;
        return 1;
    }
    inCorpusDirPath = std::string(argv[optind]);
    outCorpusDirPath = std::string(argv[optind+1]);
    if (argc - optind == 3) {
        iterNum = atoi(argv[optind+2]);
    }

    if (iterNum <= 0 || MultiNote::contourIndexLimit - 2 < iterNum) {
        std::cout << "Error: iterNum <= 0 or > "
            << MultiNote::contourIndexLimit - 2 << ": " << iterNum << std::endl;
        return 1;
    }
    
    std::cout << "doLog: " << doLog << '\n'
        << "workerNum: " << workerNum << '\n'
        << "adjacency: " << adjacency << '\n'  
        << "samplingRate: " << samplingRate << '\n'
        << "minScore: " << minScore << '\n'
        << "inCorpusDirPath: " << inCorpusDirPath << '\n'
        << "outCorpusDirPath: " << outCorpusDirPath << '\n'
        << "iterNum: " << iterNum << std::endl;

    /* open and read files */

    using namespace std::chrono;
    duration<double> oneSec = duration<double>(1.0);
    time_point<system_clock> programStartTime = system_clock::now();
    time_point<system_clock> ioStartTime = system_clock::now();

    std::string inCorpusFilePath = inCorpusDirPath + "/corpus";
    std::ifstream inCorpusFile(inCorpusFilePath, std::ios::in | std::ios::binary);
    if (!inCorpusFile.is_open()) {
        std::cout << "Failed to open corpus file: "
            << inCorpusFilePath << std::endl;
        return 1;
    }
    std::cout << "Input corpus file: " << inCorpusFilePath << std::endl;

    std::string parasFilePath = inCorpusDirPath + "/paras";
    std::ifstream parasFile(parasFilePath, std::ios::in | std::ios::binary);
    if (!parasFile.is_open()) {
        std::cout << "Failed to open parameters file: "
            << parasFilePath << std::endl;
        return 1;
    }
    std::cout << "Input parameter file: " << parasFilePath << std::endl;

    std::string outContourVocabFilePath = outCorpusDirPath + "/contour_vocab";
    std::cout << "Output contour vocab file: "
        << outContourVocabFilePath << std::endl;

    std::string outCorpusFilePath = outCorpusDirPath + "/corpus";
    std::cout << "Output merged corpus file: " << outCorpusFilePath << '\n'
        << "Reading input files" << std::endl;

    // init contours
    std::vector<Contour> contourDict;
    if (isApply) {
        std::ifstream inContourVocabFile(inContourVocabFilePath,
            std::ios::in | std::ios::binary);
        if (!inContourVocabFile.is_open()) {
            std::cout << "Failed to open input contour vocab file: "
                << inContourVocabFilePath << std::endl;
            return 1;
        }
        std::cout << "Input contour vocab path: "
            << inContourVocabFilePath << std::endl;
        contourDict = readContourVocabFile(inContourVocabFile);
        inContourVocabFile.close();
        if (contourDict.size() == 2) {
            std::cout << "Empty contour vocab file\n";
            return 0;
        }
        std::cout << "Input Contour vocab size: " << contourDict.size() - 2
            << "(+2)" << std::endl;
        if (contourDict.size() - 2 < iterNum) {
            iterNum = contourDict.size() - 2;
        }
    }
    else {
        contourDict = getDefaultContourDict();
    }

    // read parameters
    std::map<std::string, std::string> paras = readParasFile(parasFile);
    int tpq, maxDur, maxTrackNum;
    // stoi: c++11 thing
    tpq = stoi(paras[std::string("tpq")]);
    maxDur = stoi(paras[std::string("max_duration")]);
    maxTrackNum = stoi(paras[std::string("max_track_number")]);
    if (tpq <= 0 || maxDur <= 0 || maxDur > RelNote::durLimit
        || maxTrackNum <= 0) {
        std::cout << "Corpus parameter error" << '\n'
            << "tpq: " << tpq << '\n'
            << "maxDuration: " << maxDur << '\n'
            << "maxTrackNum: " << maxTrackNum << std::endl;
        return 1;
    }

    // read notes from corpus
    Corpus corpus = readCorpusFile(inCorpusFile, tpq, maxTrackNum);
    corpus.sortAllTracks();
    int numTracks = 0;
    for (int i = 0; i < corpus.pieceNum; i++) {
        numTracks += corpus.trackInstrMaps[i].size();
    }
    std::cout << "Reading done. There are " << corpus.pieceNum
        << " pieces and "  << numTracks << " tracks." << std::endl;

    size_t startMultinoteCount, multinoteCount;
    startMultinoteCount = multinoteCount = corpus.getMultiNoteCount();
    // double startMultinoteEntropy = calculateMultinoteEntropy(
    //     corpus, startMultinoteCount
    // );
    double startAvgMulpi = calculateAvgMulpiSize(corpus, false);
    double avgMulpi = startAvgMulpi;

    std::cout << "Start multinote count: " << multinoteCount
        // << ", Start multinote entropy: " << startMultinoteEntropy
        << ", Start average mulpi: " << avgMulpi
        << ", Reading used time: "
        << (system_clock::now() - ioStartTime) / oneSec
        << std::endl;

    if (multinoteCount == 0) {
        std::cout << "No notes to merge. Exited." << std::endl;
        return 1;
    }

    /* begin learning or applying */

    time_point<system_clock> iterStartTime;
    time_point<system_clock> partStartTime;
    double iterTime, findBestContourTime, mergeTime, metricsTime = 0.0;
    double totalFindBestContourTime = 0.0, totalMergeTime = 0.0;
    if (doLog) {
        if (isApply) {
            std::cout << "Iter, Avg neighbor number, "
                "Contour, Multinote count, "
                "Iteration time, Merge time" << std::endl;
        }
        else {
            std::cout << "Iter, Avg neighbor number, "
                "Found contours count, Contour, Score, Multinote count, "
                "Iteration time, Find best contour time, Merge time" << std::endl;
        }
    }
    for (int iterCount = 0; iterCount < iterNum; ++iterCount) {
        iterStartTime = system_clock::now();
        
        double avgNeighborNumber =
            (double) updateNeighbor(corpus, contourDict, tpq) / multinoteCount;

        // get contour scores and current merging contour
        Contour curContour;
        unsigned int curContourIndex = 0;
        if (isApply) {
            curContour = contourDict[iterCount + 2];
            curContourIndex = iterCount + 2;
            if (doLog) {
                std::cout << iterCount
                    << ", " << avgNeighborNumber
                    << ", \"" << contour2str(curContour) << "\", ";
            }
        }
        else {
            partStartTime = system_clock::now();
            const flat_contour_counter_t& contourCounter =
                getContourCounter(corpus, contourDict, adjacency, samplingRate);
            const std::pair<Contour, unsigned int> maxValPair =
                findMaxValPair(contourCounter);
            if (maxValPair.second <= minScore) {
                std::cout << "End early: found best score ("
                    << maxValPair.second << ") <= minScore ("
                    << minScore << ")\n";
                break;
            }
            curContour = maxValPair.first;
            if (doLog) {
                std::cout << iterCount
                    << ", " << avgNeighborNumber
                    << ", " << contourCounter.size()
                    << ", \"" << contour2str(curContour) << "\", "
                    << maxValPair.second << ", ";
            }
            curContourIndex = contourDict.size();
            contourDict.push_back(curContour);

            findBestContourTime =
                (system_clock::now() - partStartTime) / oneSec;
            totalFindBestContourTime += findBestContourTime;
        }

        // merge MultiNotes with current contour
        partStartTime = system_clock::now();
        // for each piece
        #pragma omp parallel for
        for (int i = 0; i < corpus.pieceNum; ++i) {
            // for each track
            std::vector<Track>& piece = corpus.mns[i];
            #pragma omp parallel for
            for (int j = 0; j < piece.size(); ++j) {
                Track& track = piece[j];
                // for each multinote
                for (int k = 0; k < track.size(); ++k) {
                    // for each neighbor
                    for (int n = 1; n <= track[k].neighbor; ++n) {
                        if (k + n >= track.size()) {
                            continue;
                        }
                        if (track[k].vel == 0
                            || track[k+n].vel == 0
                            || track[k].vel != track[k+n].vel) {
                            continue;
                        }
                        Contour s = getContourOfMultiNotePair(
                            track[k],
                            track[k+n],
                            contourDict
                        );
                        if (s == curContour) {
                            // change left multinote to merged multinote
                            // because relnote & multinote are sorted the same
                            // the first relnote in the new contour is the
                            // first relnote in left multinote's contour
                            uint8_t newStretch =
                                contourDict[track[k].contourIndex][0].relDur
                                * track[k].stretch / curContour[0].relDur;
                            // unit cannot be greater than max_duration
                            if (newStretch > maxDur) continue;
                            track[k].stretch = newStretch;
                            track[k].contourIndex = curContourIndex;

                            // mark right multinote to be removed by vel = 0
                            track[k+n].vel = 0;
                            break;
                        }
                    }
                }
                // remove multinotes with vel == 0
                track.erase(
                    std::remove_if(
                        track.begin(),
                        track.end(),
                        [] (const MultiNote& m) {
                            return m.vel == 0;
                        }
                    ),
                    track.end()
                );
            }
        }
        mergeTime = (system_clock::now() - partStartTime) / oneSec;
        totalMergeTime += mergeTime;
        iterTime = (system_clock::now() - iterStartTime) / oneSec;
        if (doLog) {
            // exclude the time used on calculating metrics
            partStartTime = system_clock::now();
            multinoteCount = corpus.getMultiNoteCount();
            metricsTime += (system_clock::now() - partStartTime) / oneSec;
            std::cout << multinoteCount << ", " << iterTime;
            if (!isApply) {
                std::cout << ", " << findBestContourTime;
            }
            std::cout << ", " << mergeTime << std::endl;
        }
    }

    if (!doLog) {
        multinoteCount = corpus.getMultiNoteCount();
    }
    // double endMultinoteEntropy = calculateMultinoteEntropy(
    //     corpus, multinoteCount
    // );
    avgMulpi = calculateAvgMulpiSize(corpus);
    std::cout << "End multinote count: " << multinoteCount
        // << ", End multinote entropy: " << endMultinoteEntropy
        << ", End average mulpi: " << avgMulpi
        << ", Total find bset contour time: " << totalFindBestContourTime
        << ", Total merge time: " << totalMergeTime
        << std::endl;

    // open/create and write files

    std::ofstream outContourVocabFile(
        outContourVocabFilePath,
        std::ios::out | std::ios::trunc
    );
    if (!outContourVocabFile.is_open()) {
        std::cout << "Failed to open vocab output file: "
            << outContourVocabFilePath << std::endl;
        return 1;
    }
    std::ofstream outCorpusFile(
        outCorpusFilePath,
        std::ios::out | std::ios::trunc
    );
    if (!outCorpusFile.is_open()) {
        std::cout << "Failed to open merged corpus output file: "
            << outCorpusFilePath << std::endl;
        return 1;
    }

    ioStartTime = system_clock::now();
    writeContourVocabFile(outContourVocabFile, contourDict);
    std::cout << "Writing merged corpus file" << std::endl;
    writeOutputCorpusFile(outCorpusFile, corpus, contourDict, maxTrackNum);
    time_point<system_clock> ioEndTime = system_clock::now();
    std::cout << "Writing done. Writing used time: "
        << (ioEndTime - ioStartTime) / oneSec << '\n'
        << "Total used time: "
        << (ioEndTime - programStartTime) / oneSec - metricsTime
        << std::endl;
    return 0;
}
