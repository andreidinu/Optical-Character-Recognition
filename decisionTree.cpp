// copyright - code - Andrei Dinu, Stefan Hermeniuc
// copyright - scheleton - Luca Istrate, Andrei Medar

#include "./decisionTree.h"  // NOLINT(build/include)
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>

using std::string;
using std::pair;
using std::vector;
using std::unordered_map;
using std::make_shared;
using std::mt19937;

// structura unui nod din decision tree
// splitIndex = dimensiunea in functie de care se imparte
// split_value = valoarea in functie de care se imparte
// is_leaf si result sunt pentru cazul in care avem un nod frunza
Node::Node() {
    is_leaf = false;
    left = nullptr;
    right = nullptr;
}

void Node::make_decision_node(const int index, const int val) {
    split_index = index;
    split_value = val;
}

void Node::make_leaf(const vector<vector<int>> &samples,
                     const bool is_single_class) {
    // Seteaza nodul ca fiind de tip frunza (modificati is_leaf si result)
    // is_single_class = true -> toate testele au aceeasi clasa (acela e result)
    // is_single_class = false -> se alege clasa care apare cel mai des
    this->is_leaf = true;
    if (is_single_class) {
    	this->result = samples[0][0];
    } else {
    	int i;
    	int max = 0;
    	int max_ind = 0;
    	vector<int> v(10);

    	for(i = 0; i < samples.size(); i++) {
    		++v[samples[i][0]];
    	}
    	for(i = 0; i < v.size(); i++) {
    		if (max < v[i]) {
    			max = v[i];
    			max_ind = i;
    		}
    	}
    	this->result = max_ind;
    }
}

pair<int, int> find_best_split(const vector<vector<int>> &samples,
                               const vector<int> &dimensions) {
    // Intoarce cea mai buna dimensiune si valoare de split dintre testele
    // primite. Prin cel mai bun split (dimensiune si valoare)
    // ne referim la split-ul care maximizeaza IG
    // pair-ul intors este format din (split_index, split_value)

    int ds = dimensions.size();
    float maxIG = -1;
    int splitIndex = -1, splitValue = -1;

    for (int i = 0; i < ds; i++) {
        // Calculam split-ul considerand split index dimensions[i] si split
        // value media elementelor unice de pe coloana decisions[i] din samples
        int sum = 0;

        vector<int> col = compute_unique(samples, dimensions[i]);
        int  col_s = col.size();

        for (int j = 0; j < col_s; j++) {
            sum += col[j];
        }

        int mean = sum / col_s;

        pair<vector<vector<int>>, vector<vector<int>>> subspl =
        split(samples, dimensions[i], mean);

        // Calculam Information Gain pentru fiecare split valid si aflam maximul
        if (!subspl.first.empty() && !subspl.second.empty()) {
            float parent_entropy = get_entropy(samples);
            float left_entropy = get_entropy(subspl.first);
            int nl = subspl.first.size();
            float right_entropy = get_entropy(subspl.second);
            int nr = subspl.second.size();

            float IG = parent_entropy - (nl * left_entropy + nr * right_entropy)
            / (nl + nr);

            if (IG > maxIG) {
                maxIG = IG;
                splitIndex = dimensions[i];
                splitValue = mean;
            }
        }
    }

    return pair<int, int>(splitIndex, splitValue);
}

void Node::train(const vector<vector<int>> &samples) {
    // Antreneaza nodul curent si copii sai, daca e nevoie
    // 1) verifica daca toate testele primite au aceeasi clasa (raspuns)
    // Daca da, acest nod devine frunza, altfel continua algoritmul.
    // 2) Daca nu exista niciun split valid, acest nod devine frunza. Altfel,
    // ia cel mai bun split si continua recursiv

    if (same_class(samples)) {
        this->make_leaf(samples, true);
    } else {
        int size = samples[0].size();
        // Generam random un vector decisions si il folosim ca sa aflam cel
        // mai bun split
        vector<int> dimensions = random_dimensions(size);
        pair<int, int> p = find_best_split(samples, dimensions);

        // Daca nu s-a gasit niciun split valid facem nodul frunza
        if (p.first == -1 && p.second == -1) {
            this->make_leaf(samples, false);
            return;
        }

        this->split_index = p.first;
        this->split_value = p.second;

        // Facem split-ul si continuam recursiv
        pair<vector<vector<int>>, vector<vector<int>>> my_pair =
        split(samples, p.first, p.second);

        left = make_shared<Node>(Node());
        right = make_shared<Node>(Node());

        left->train(my_pair.first);
        right->train(my_pair.second);
    }
}

int Node::predict(const vector<int> &image) const {
    // Intoarce rezultatul prezis de catre decision tree;
    if (!this->is_leaf) {
        if ((image[split_index - 1]) > split_value) {
            return right->predict(image);
        } else {
            return left->predict(image);
        }
    }

    return this->result;
}

bool same_class(const vector<vector<int>> &samples) {
    // Verifica daca testele primite ca argument au toate aceeasi
    // clasa(rezultat). Este folosit in train pentru a determina daca
    // mai are rost sa caute split-uri
    vector<int> v;

    // Clasa este data de elementul de pe prima pozitie
    for (auto elem : samples) {
        v.push_back(elem[0]);
    }
    // Sortam vectorul de clase si aflam daca exista duplicate
    std::sort(v.begin(), v.end());

    for (int i = 1; i < v.size(); i++) {
        if (v[i] != v[i - 1]) {
            return false;
        }
    }
    return true;
}

float get_entropy(const vector<vector<int>> &samples) {
    // Intoarce entropia testelor primite
    assert(!samples.empty());
    vector<int> indexes;

    int size = samples.size();
    for (int i = 0; i < size; i++) indexes.push_back(i);

    return get_entropy_by_indexes(samples, indexes);
}

float get_entropy_by_indexes(const vector<vector<int>> &samples,
                             const vector<int> &index) {
    // Intoarce entropia subsetului din setul de teste total(samples)
    // Cu conditia ca subsetul sa contina testele ale caror indecsi se gasesc in
    // vectorul index (Se considera doar liniile din vectorul index)
    int index_size = index.size();
    float entropy = 0.0;
    vector<float> recc(10, 0);

    for (int i = 0; i < index_size; i++) {
        recc[samples[index[i]][0]]++;
    }

    for (int i = 0; i < 10; i++) {
        float pi = (float) recc[i] / (float) index_size;
        if (pi) {
            entropy -= pi * log2(pi);
        }
    }

    return entropy;
}

vector<int> compute_unique(const vector<vector<int>> &samples, const int col) {
    // Intoarce toate valorile (se elimina duplicatele)
    // care apar in setul de teste, pe coloana col
    vector<int> uniqueValues;

    int n = samples.size();
    for (int i = 0; i < n; i++) {
        uniqueValues.push_back(samples[i][col]);
    }

    std::sort(uniqueValues.begin(), uniqueValues.end());

    vector<int>::iterator it;
    it = std::unique(uniqueValues.begin(), uniqueValues.end());

    uniqueValues.resize(std::distance(uniqueValues.begin(), it));

    return uniqueValues;
}

pair<vector<vector<int>>, vector<vector<int>>> split(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // Intoarce cele 2 subseturi de teste obtinute in urma separarii
    // In functie de split_index si split_value
    vector<vector<int>> left, right;

    auto p = get_split_as_indexes(samples, split_index, split_value);
    for (const auto &i : p.first) left.push_back(samples[i]);
    for (const auto &i : p.second) right.push_back(samples[i]);

    return pair<vector<vector<int>>, vector<vector<int>>>(left, right);
}

pair<vector<int>, vector<int>> get_split_as_indexes(
    const vector<vector<int>> &samples, const int split_index,
    const int split_value) {
    // Intoarce indecsii sample-urilor din cele 2 subseturi obtinute in urma
    // separarii in functie de split_index si split_value
    vector<int> left, right;
    int size = samples.size();
    for (int i = 0; i < size; i++) {
        if (samples[i][split_index] <= split_value) {
            left.push_back(i);
        }
    }

    for (int j = 0; j < size; j++) {
        if (samples[j][split_index] > split_value) {
            right.push_back(j);
        }
    }
    return make_pair(left, right);
}

vector<int> random_dimensions(const int size) {
    // Intoarce sqrt(size) dimensiuni diferite pe care sa caute splitul maxim
    // Precizare: Dimensiunile gasite sunt > 0 si < size
    vector<int> dim;
	int rad = floor(sqrt(size));

    int random_integer;
    std::random_device rd;
    std::mt19937 rng(rd());

    vector<bool> added(size, false);
    added[0] = true;

    // Generam dimensiuni random pana cand obtinem floor(sqrt(size)) dimensiuni
    // unice si nenule
	for (int i = 0; i < rad; ++i) {
        int number = rng() % size;
        while (added[number]) {
            number = rng() % size;
        }
		dim.push_back(number);
        added[number] = true;
	}

	return dim;
}
