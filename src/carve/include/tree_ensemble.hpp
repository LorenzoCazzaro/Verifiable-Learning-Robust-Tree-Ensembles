#pragma once

#include "tree.hpp"

#include <algorithm>
#include <map>
#include <list>

namespace vl {

struct tree_ensemble {
    tree_ensemble() {}

    void parse(std::ifstream& in) {
        std::string line;
        std::getline(in, line);
        std::istringstream is(line);
        is >> line;
        assert(line == "classifier-forest");
        uint32_t num_trees;
        is >> num_trees;
        std::cout << "num_trees = " << num_trees << std::endl;
        m_trees.resize(num_trees);
        for (uint32_t i = 0; i != num_trees; ++i) {
            // std::cout << "parsing " << i << "-th tree" << std::endl;
            m_trees[i].parse(in);
        }
    }

    //predict the label of an instance by obtaining the predictions from each tree of the ensemble and then using boyer_moore majority voting
    label_t predict(instance_t const& x) const {
        std::vector<label_t> labels;
        labels.reserve(num_trees());
        for (auto const& t : m_trees) labels.push_back(t.predict(x));
        label_t candidate = boyer_moore_majority_voting(labels);
        if (candidate == constants::invalid_label) {
            /* ensemble did not reach a consensus */
            return labels.front(); // return label given by first tree
        }
        return candidate;
    }

    bool robust(instance_t const& x, label_t y, float p, float k) const {
        if (predict(x) == y) return stable(x, y, p, k);
        return false;
    }

    //annotate each tree of the ensemble
    void annotate() {
        for (auto& t : m_trees) t.annotate();
    }

    //access a tree of the ensemble by index
    tree const& operator[](uint32_t i) const {
        assert(i < m_trees.size());
        return m_trees[i];
    }

    uint32_t num_trees() const { return m_trees.size(); }

    uint32_t num_features() const {
        assert(!m_trees.empty());
        return m_trees.front().num_features();
    }

    void print(std::ostream& out){
        out << "Print tree ensemble, contains " << m_trees.size() << " trees" << endl;
        for (auto& t : m_trees) t.print(out);
    }

    //check the stability of the tree ensemble on the instance x with label y under the attack A_{p,k}
    bool stable(instance_t const& x, label_t y, float p, float k) const {
        const uint32_t m = num_trees();
        const uint32_t d = num_features();

        struct perturbation {
            perturbation() : norm(constants::inf) {}
            double norm;
            std::vector<float_t> delta;
        };

        std::vector<perturbation> D;
        D.resize(m);

        uint32_t num_unstable_trees = 0;

        for (uint32_t i = 0; i != m; ++i) {
            auto const& t = m_trees[i];
            auto L = t.reachable(x, p, k);

            uint32_t arg_min = L.size();

            double min_norm_p = constants::inf;
            for (uint32_t i = 0; i != L.size(); ++i) {
                auto const& mp = L[i];
                if (mp.label != y) {
                    if (mp.norm < min_norm_p) {
                        min_norm_p = mp.norm;
                        arg_min = i;
                    }
                }
            }

            if (arg_min < L.size()) {  // it exists
                D[i].norm = L[arg_min].norm;
                D[i].delta = std::move(L[arg_min].delta);
                ++num_unstable_trees;
            }
        }

        //compose the result (following the algorithm described in the paper)
        if (num_unstable_trees >= m / 2 + 1) {
            std::sort(D.begin(), D.end(),
                      [](auto const& x, auto const& y) { return x.norm < y.norm; });
            instance_t delta(d, 0.0);
            for (uint32_t i = 0; i != m / 2 + 1; ++i) {
                for (uint32_t j = 0; j != d; ++j) delta[j] += D[i].delta[j];
            }
            if (norm(delta, p) <= k) return false;
        }

        return true;
    }

    //return a map with features as key and lists of thresholds per feature as values of the tree ensemble
    void get_thresholds(std::map<uint32_t, list<float>>& threshold_map){
        for(auto iter = this->m_trees.begin(); iter != this->m_trees.end(); iter++){
            iter->get_thresholds_from_root(threshold_map);
        }
    }

private:
    std::vector<tree> m_trees;
};

}
