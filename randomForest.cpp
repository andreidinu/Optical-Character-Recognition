// copyright - code - Andrei Dinu, Stefan Hermeniuc
// copyright - scheleton - Luca Istrate, Andrei Medar
#include "randomForest.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include "decisionTree.h"

using std::vector;
using std::pair;
using std::string;
using std::mt19937;
using std::unordered_map;

vector<vector<int>> get_random_samples(const vector<vector<int>> &samples,
                                       int num_to_return) {
    // Intoarce un vector de marime num_to_return cu elemente random,
    // diferite din samples
    vector<vector<int>> ret;
    vector<int> rand;
    int mat_size = samples.size();
    int random_integer;
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, mat_size - 1);
    unordered_map<int, bool> rand_table;

    while (rand.size() != mat_size) {
    	random_integer = uni(rng);
    	if (!rand_table[random_integer]) {
    		rand.push_back(random_integer);
            rand_table[random_integer] = true;
    	}
    }

    for (int i = 0; i < num_to_return; ++i) {
    	ret.push_back(samples[i]);
    }

    return ret;
}

RandomForest::RandomForest(int num_trees, const vector<vector<int>> &samples)
    : num_trees(num_trees), images(samples) {}

void RandomForest::build() {
    // Aloca pentru fiecare Tree cate n / num_trees
    // Unde n e numarul total de teste de training
    // Apoi antreneaza fiecare tree cu testele alese
    assert(!images.empty());
    vector<vector<int>> random_samples;

    int data_size = images.size() / num_trees;

    for (int i = 0; i < num_trees; i++) {
        // cout << "Creating Tree nr: " << i << endl;
        random_samples = get_random_samples(images, data_size);

        // Construieste un Tree nou si il antreneaza
        trees.push_back(Node());
        trees[trees.size() - 1].train(random_samples);
    }
}

int RandomForest::predict(const vector<int> &image) {
    // Va intoarce cea mai probabila prezicere pentru testul din argument
    // se va interoga fiecare Tree si se va considera raspunsul final ca
    // fiind cel majoritar
    int s_class;
    int max = 0;
    vector<int> digit(10);

    for (int i = 0; i < num_trees; i++) {
        s_class = trees[i].predict(image);
        digit[s_class]++;
    }

    for (int i = 0; i < 10; i++) {
        if (max < digit[i]) {
            max = digit[i];
            s_class = i;
        }
    }
    return s_class;
}
