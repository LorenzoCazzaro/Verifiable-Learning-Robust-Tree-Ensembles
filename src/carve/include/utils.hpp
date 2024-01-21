#pragma once

#include <vector>
#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

#include <iostream>

using namespace std;

namespace vl {

typedef uint32_t feature_t;
typedef float float_t;
typedef int label_t;

typedef std::vector<float_t> instance_t;

namespace constants {
constexpr feature_t invalid_feature = feature_t(-1);
constexpr float_t invalid_threshold = float_t(-1);
constexpr label_t invalid_label = label_t(-1);
constexpr float_t inf = std::numeric_limits<float_t>::max();
}  // namespace constants

struct hyper_rectangle {
    hyper_rectangle() : label(constants::invalid_label) {}
    label_t label;
    std::vector<std::pair<float_t, float_t>> H;
    bool empty = false;
    void set_empty() {
        for (uint32_t i = 0; i < H.size(); i++) {
            if (H[i].first >= H[i].second) {
                empty = true;
                return;
            }
        }
    }
};

struct min_perturbation {
    min_perturbation() : label(constants::invalid_label) {}
    label_t label;
    double norm;
    std::vector<float_t> delta;
};

double norm(instance_t const& x, float p) {
    if (p != constants::inf) {
        double ret = 0.0;
        for (auto x_i : x) ret += pow(abs(x_i), p);
        return pow(ret, 1.0 / p);
    } else {
        double ret = 0.0;
        for (auto x_i : x) ret = std::max<double>(abs(x_i), ret);
        return ret;
    }
}

// compute the most voted label using boyer_moore majority voting algorithm
label_t boyer_moore_majority_voting(std::vector<label_t> const& labels) {
    label_t candidate = constants::invalid_label;

    for (uint32_t i = 0, num_votes = 0; i != labels.size(); ++i) {
        if (num_votes == 0) {
            candidate = labels[i];
            num_votes = 1;
        } else {
            if (labels[i] == candidate) {
                ++num_votes;
            } else {
                --num_votes;
            }
        }
    }

    uint32_t num_instances = 0;
    for (auto l : labels) {
        if (l == candidate) ++num_instances;
    }

    if (num_instances >= labels.size() / 2 + 1) return candidate;

    return constants::invalid_label;
}

}  // namespace vl