#include <iostream>
#include <random>
#include <chrono>
#include <sys/resource.h>
#include <cstring>

#include "tree_ensemble.hpp"
#include "parser.hpp"

using namespace vl;

typedef std::chrono::high_resolution_clock clock_type;

void parse_row_in_csv(std::ifstream& in, std::string& line, instance_t& x, label_t& y) {
    std::getline(in, line);
    std::istringstream is(line);
    std::string str;
    x.clear();
    std::getline(is, str, ',');
    y = std::stof(str);
    while (std::getline(is, str, ',')) x.push_back(std::stof(str));
}

int main(int argc, char** argv) {
    cmd_line_parser::parser parser(argc, argv);
    parser.add("model_filename", "Model filename: must be a .silva file.", "-i", true);
    parser.add("testset_filename", "Testset filename: must be a .csv file.", "-t", true);
    parser.add("p", "Determines the norm used.", "-p", true);
    parser.add("k", "Determines the strength of the attacker.", "-k", true);
    parser.add("index_of_instance", "Index of the instance of the dataset to verify.", "-ioi", -1);
    if (!parser.parse()) return 1;

    auto model_filename = parser.get<std::string>("model_filename");
    auto testset_filename = parser.get<std::string>("testset_filename");

    float p = 0.0;
    if (parser.get<std::string>("p") == "inf") {
        p = constants::inf;
    } else {
        p = parser.get<float>("p");
    }
    cout << "p: " << p << endl;

    auto k = parser.get<float>("k");
    cout << "k: " << k << endl;

    int ioi = parser.get<float>("index_of_instance");
    cout << "Index of instance on which verify the model: " << ioi << endl;

    tree_ensemble T;

    {
        /* 1. load the model */
        auto start = clock_type::now();
        std::ifstream in(model_filename);
        T.parse(in);
        in.close();
        auto stop = clock_type::now();
        std::cout << "1. loading model: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
                  << " [msec]" << std::endl;
    }
    {
        /* 2. annotate */
        auto start = clock_type::now();
        T.annotate();
        auto stop = clock_type::now();
        std::cout << "2. annotate: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
                  << " [msec]" << std::endl;
    }
    {
        /* 3. test */
        std::ifstream in(testset_filename);
        std::string line;
        instance_t x;
        label_t y;

        std::getline(in, line);
        std::istringstream is(line);
        uint32_t num_queries = 0;
        uint32_t num_features = 0;
        is >> line;  // skip #
        is >> num_queries;
        if (ioi >= 0) num_queries = 1;
        is >> num_features;
        assert(num_features == T.num_features());
        x.reserve(num_features);

        vector<instance_t> queries;
        vector<label_t> labels;
        if (ioi == -1) {
            for (uint32_t i = 0; i != num_queries; ++i) {
                parse_row_in_csv(in, line, x, y);
                queries.push_back(x);
                labels.push_back(y);
            }
        } else {
            for (uint32_t i = 0; i != ioi; ++i) { std::getline(in, line); }
            parse_row_in_csv(in, line, x, y);
            queries.push_back(x);
            labels.push_back(y);
        }
        in.close();
        std::cout << "performing " << num_queries << " queries..." << std::endl;
        // added variables to check the consistency with SILVA results
        uint32_t n_correct = 0;
        uint32_t n_stable = 0;
        uint32_t n_unstable = 0;
        uint32_t n_robust = 0;
        uint32_t n_fragile = 0;
        double total_verification_time = 0.0;
        for (uint32_t i = 0; i != num_queries; ++i) {
            auto start_x_sample = clock_type::now();
            x = queries[i];
            y = labels[i];
            label_t y_pred = T.predict(x);

            bool same_prediction = y_pred == y;
            bool stable = T.stable(x, y_pred, p, k);
            n_correct += same_prediction;
            n_stable += stable;
            n_unstable += !stable;
            n_robust += same_prediction && stable;
            n_fragile += same_prediction && !stable;
            auto end_x_sample = clock_type::now();
            double verification_time_x_instance =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_x_sample - start_x_sample)
                    .count();
            total_verification_time += verification_time_x_instance;
            cout << "query " << i << ": ";
            cout << "pred label " << y_pred << ", true label " << y << ", STATUS: ";
            cout << (stable ? (same_prediction ? "ROBUST" : "VULNERABLE")
                            : (same_prediction ? "FRAGILE" : "BROKEN"))
                 << endl;
            cout << "Time required per query " << i << ": " << verification_time_x_instance
                 << " [msec]" << endl;
        }

        std::cout << "3. test " << num_queries << " queries: " << total_verification_time
                  << " [msec] (" << total_verification_time / num_queries << " msec/query)"
                  << std::endl;

        std::cout << "n queries correctly classified: " << n_correct << std::endl;

        std::cout << "accuracy: " << (n_correct * 100.0) / num_queries << "%" << std::endl;

        std::cout << "robustness: " << (n_robust * 100.0) / num_queries << "%" << std::endl;

        std::cout << "n robust queries: " << n_robust << std::endl;

        std::cout << "n fragile queries: " << n_fragile << std::endl;

        std::cout << "n vulnerable queries: " << n_stable - n_robust << std::endl;

        std::cout << "n broken queries: " << n_unstable - n_fragile << std::endl;
    }

    return 0;
}
