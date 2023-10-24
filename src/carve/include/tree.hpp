#pragma once

#include "utils.hpp"

#include <map>
#include <list>

namespace vl {

struct tree {
    tree() : m_num_leaves(0), m_num_features(0), m_num_classes(0) {}

    void parse(std::ifstream& in) {
        std::string line;
        std::getline(in, line);
        std::istringstream is(line);
        is >> line;
        assert(line == "classifier-decision-tree");
        is >> m_num_features;
        is >> m_num_classes;
        std::getline(in, line);  // skip line containing labels
        m_root = new node;
        uint32_t max_depth = parse_node(in, m_root, 1);  // parse root
        // std::cout << "max_depth = " << max_depth << std::endl;
        // std::cout << "m_num_leaves = " << m_num_leaves << std::endl;
        // std::cout << "m_num_features = " << m_num_features << std::endl;
        // std::cout << "m_num_classes = " << m_num_classes << std::endl;
        (void)max_depth;
    }

    ~tree() { delete_node(m_root); }

    struct node {
        node()
            : feature(constants::invalid_feature)
            , threshold(constants::invalid_threshold)
            , label(constants::invalid_label)
            , left(nullptr)
            , right(nullptr) {}
        feature_t feature;
        float_t threshold;
        label_t label;
        node* left;
        node* right;
    };

    bool annotated() const { return !m_hyper_rectangles.empty(); }

    void annotate() {
        if (!annotated()) {
            m_hyper_rectangles.reserve(m_num_leaves);
            hyper_rectangle root_hr;
            root_hr.H.resize(m_num_features, {-constants::inf, constants::inf});
            annotate(m_root, root_hr);  // annotate recursively from root
            assert(m_hyper_rectangles.size() == m_num_leaves);
        }
    }

    label_t predict(instance_t const& x) const { return predict(x, m_root); }

    //use the hyperrectangles that annotates the leaves to find the perturbations that allows the instance x to reach the leaves of the tree
    std::vector<min_perturbation> reachable(instance_t const& x, float p, float k) const {
        
        assert(x.size() == m_num_features);
        std::vector<min_perturbation> L;
        L.reserve(m_num_leaves);

        for (auto const& hr : m_hyper_rectangles) { //consider all the hyperrectangles that annotate the leaves of the tree
            if(!hr.empty) {
                assert(hr.H.size() == m_num_features);
                min_perturbation mp;
                mp.label = hr.label;
                mp.delta.resize(m_num_features, 0.0);
                for (uint32_t i = 0; i != m_num_features; ++i) {
                    auto l_i = hr.H[i].first;
                    auto r_i = hr.H[i].second;
                    auto x_i = x[i];
                    //compute the perturbation mp as described in the paper
                    if (x_i <= l_i) {
                        mp.delta[i] = std::nextafterf(l_i - x_i, constants::inf);
                    } else if (x_i > r_i) {
                        mp.delta[i] = r_i - x_i;
                    }
                }

                mp.norm = norm(mp.delta, p);
                if (mp.norm <= k) L.push_back(mp); //the perturbation allows to reach the leaf only if its norm is less than or equal k
            }
        }

        return L;
    }

    uint32_t num_features() const { return m_num_features; }


    void print(std::ostream& out) {
        out << "Print decision tree, num features " << m_num_features << ", num classes " << m_num_classes << endl;
        print_aux(m_root, out, "");
    }

    void get_thresholds_from_root(std::map<uint32_t, list<float>>& threshold_map){
        get_thresholds(this->m_root, threshold_map);
    } 

private:
    uint32_t m_num_leaves;
    uint32_t m_num_features;
    int m_num_classes;
    node* m_root;
    std::vector<hyper_rectangle> m_hyper_rectangles; //set of hyperrectangles obtained after the annotation of the leaves of the tree

    bool is_leaf(node const* n) const { return n->left == nullptr and n->right == nullptr; }

    //predict the label given the instance and the root of the tree n
    label_t predict(instance_t const& x, node const* n) const {
        if (is_leaf(n)) return n->label;
        if (x[n->feature] <= n->threshold) return predict(x, n->left);
        return predict(x, n->right);
    }

    //annotate the leaves of the tree whose radix is n
    void annotate(node const* n, hyper_rectangle& parent_hr) {
        if (is_leaf(n)) {
            parent_hr.label = n->label;
            parent_hr.set_empty();
            m_hyper_rectangles.push_back(parent_hr);
            return;
        }

        hyper_rectangle l_hr = parent_hr;
        hyper_rectangle r_hr = parent_hr;
        feature_t feature = n->feature;
        //build the hyperrectangle from the parent hyperrectangles by limiting the interval on the feature using the threshold of the considered node
        l_hr.H[feature].second = std::min(l_hr.H[feature].second, n->threshold);
        r_hr.H[feature].first = std::max(r_hr.H[feature].first, n->threshold);
        annotate(n->left, l_hr);
        annotate(n->right, r_hr);
    }

    uint32_t parse_node(std::ifstream& in, node* n, uint32_t depth) {
        assert(n->feature == constants::invalid_feature);
        assert(n->threshold == constants::invalid_threshold);
        assert(n->label == constants::invalid_label);

        std::string line;
        std::getline(in, line);
        std::istringstream is(line);
        std::string node_type;
        is >> node_type;
        uint32_t max_depth = depth;

        /* internal node*/
        if (node_type == "SPLIT") {
            feature_t feature;
            is >> feature;
            assert(feature < m_num_features);
            float_t threshold;
            is >> threshold;
            n->feature = feature;
            n->threshold = threshold;
            n->left = new node;
            n->right = new node;
            uint32_t l_depth = parse_node(in, n->left, depth + 1);
            uint32_t r_depth = parse_node(in, n->right, depth + 1);
            max_depth = std::max(l_depth, r_depth);
        }
        /* leaf */
        else if (node_type == "LEAF") {
            label_t label = 0;
            uint32_t max = 0;
            label_t pos = 0;
            while (is) {
                uint32_t val;
                is >> val;
                if (val > max) {
                    max = val;
                    label = pos;
                }
                ++pos;
            }
            assert(label < m_num_classes);
            n->label = label;
            ++m_num_leaves;
        }

        return max_depth;
    }

    void delete_node(node const* n) {
        if (!is_leaf(n)) {
            delete_node(n->left);
            delete_node(n->right);
        }
        delete n;
    }

    void print_aux(node* u, std::ostream& out, string identation_str) {
        if(u) {
            if(u->left || u->right) {
                out << identation_str + "INTERNAL NODE: " << u->feature << " <= " << u->threshold << endl;
                print_aux(u->left, out, identation_str+"\t");
                print_aux(u->right, out, identation_str+"\t");
            } else {
                out << identation_str + "LEAF NODE: " << u->label << endl;
            }
        }
    }

    void get_threshold(node const* n, std::map<uint32_t, list<float>>& threshold_map){
        threshold_map[n->feature].push_back(n->threshold);
    }

    //return a map with features as key and lists of thresholds per feature as values of the tree
    void get_thresholds(node const* n, std::map<uint32_t, list<float>>& threshold_map){
        if (!is_leaf(n)) {
            get_threshold(n, threshold_map);
            if (n->left)
                get_thresholds(n->left, threshold_map);
            if (n->right)
                get_thresholds(n->right, threshold_map);
        }
    }
};

}  // namespace vl