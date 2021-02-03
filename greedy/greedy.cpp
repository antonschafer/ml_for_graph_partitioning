#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <tuple>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

using namespace std;

vector<tuple<int, int>>* readGraph(string fname, int* n){
	ifstream inputFile(fname);
	int a;
	int m;
	int weightedType;
	bool weighted = false;
	string line;


	if (inputFile.is_open()) {

		getline(inputFile, line);
 		istringstream iss(line);
		iss >> m;
		iss >> *n;

		vector<tuple<int, int>> *adjList = new vector<tuple<int, int>>[*n];

		int found = 0;

		if(iss >> weightedType) {
			weighted = true;
			printf("Not implemented");
			exit(1);
		} else {
			int u;
			int v;
			while (found < m) {
				if(getline(inputFile, line)) {
					found ++;
				} else {
					printf("Invalid input");
					exit(1);
				}
 				istringstream lineStream(line);
				lineStream >> u;
				lineStream >> v;

				u --; // make 0 indexed
				v --; // make 0 indexed

				tuple<int, int>* entryV = new tuple<int, int>(u, 1);
				tuple<int, int>* entryU = new tuple<int, int>(v, 1);

				adjList[u].push_back(*entryU);
				adjList[v].push_back(*entryV);
			}
		}
		inputFile.close();
		return adjList;
	} else {
		printf("Could not open file");
		exit(1);
	}
}



int main(int argc, char *argv[]) {
	if(argc != 3) {
		printf("Please provide a graph file and the number of partitions\n");
		exit(1);
	}
	string fname = argv[1];
	int k = stoi(argv[2]);
	int n = 0;


	vector<tuple<int, int>> *adjList = readGraph(fname, &n);

	/*
	printf("%d\n", n);
	for(int i = 0; i<n; i++) {
		for(int j = 0; j<adjList[i].size(); j++) {
			printf("%d, %d: %d\n", i, get<0>(adjList[i][j]), get<1>(adjList[i][j]));
		}
	}
	*/

	int node_2_partition[n];
	for(int i = 0; i<n; i++) {
		node_2_partition[i] = i;
	}

  	srand (time(NULL));
	// shuffle and generate random partitioning
	for(int i = 0; i<n; i++) {
		int r = i + rand()%(n-i);
		int partition = node_2_partition[r] % k;
		node_2_partition[r] = node_2_partition[i];

		// assign partitiong
		node_2_partition[i] = partition;
	}

	long cut_weight = 0;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j<adjList[i].size(); j++) {
			int u = i;
			int v = get<0>(adjList[i][j]);
			int c = get<1>(adjList[i][j]);


			if(node_2_partition[u] != node_2_partition[v]) {
				cut_weight += c;
			}
		}
	}	
	cut_weight /= 2;  // account for edges counted twice

	int u;
	int v;
	long best_swap;


	// TODO precompute diff values and use priority queue to speed up a lot (asymptotically)
	do {
		best_swap = 0;
		for(int i = 0; i<n; i++) {
			for(int j = 0; j<i; j++) {
				int partI = node_2_partition[i];
				int partJ = node_2_partition[j];
				if(partI == partJ) continue;

				long diff = 0;
				for(int l = 0; l < adjList[i].size(); l++) {
					int neighbor = get<0>(adjList[i][l]);
					if(neighbor == j) continue;

					int c = get<1>(adjList[i][l]);

					if(node_2_partition[neighbor] == partI) diff -= c;
					if(node_2_partition[neighbor] == partJ) diff += c;
				}
				for(int l = 0; l < adjList[j].size(); l++) {
					int neighbor = get<0>(adjList[j][l]);
					if(neighbor == i) continue;

					int c = get<1>(adjList[j][l]);

					if(node_2_partition[neighbor] == partJ) diff -= c;
					if(node_2_partition[neighbor] == partI) diff += c;
				}

				// update best
				if(diff > best_swap) {
					best_swap = diff;
					u = i;
					v = j;
				}
			}
		}


		if(best_swap <= 0){
			break;
		}


		int partU = node_2_partition[u];
		node_2_partition[u] = node_2_partition[v];
		node_2_partition[v] = partU;
		cut_weight -= best_swap;


	} while(best_swap > 0);



	printf("%ld\n", cut_weight);
}




