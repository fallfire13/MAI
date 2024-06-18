#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

int n1; //количество вершин в первой доле графа
int n2; //количество вершин во второй доле графа
int m; //количество ребер в графе
vector<int> *adj; //список смежности
vector<int> used; //массив для хранения информации о пройденных и не пройденных вершинах
int mtSize = 0; //размер максимального паросочетания
vector<int> mt; //массив для хранения ребер, образующих максимальное паросочетание

//алгоритм Куна поиска максимального паросочетания
bool kuhn(int v) {
//если вершина является пройденной, то не производим из нее вызов процедуры
	if (used[v]) {
		return false;
	}
	used[v] = true; //помечаем вершину первой доли, как пройденную
//просматриваем все вершины второй доли, смежные с рассматриваемой вершиной первой доли
	for (int i = 0; i < adj[v].size(); ++i){ 
		int w = adj[v][i]; //нашли увеличивающую цепь, добавляем ребро (v, w) в паросочетание 
			if (mt[w] == -1 || kuhn(mt[w])) { 
				mt[w] = v; return true; 
			} 
		} 
	return false; 
} //процедура считывания входных данных с консоли 
void readData() { //считываем количество вершин в первой и второй доли и количество ребер графа 
	scanf("%d %d %d", &n1, &n2, &m); //инициализируем список смежности размерности n1 
		adj = new vector<int>[n1];

	//считываем граф, заданный списком ребер
	for (int i = 0; i < m; ++i) { 
		int v, w; scanf("%d %d", &v, &w); v--; w--; //добавляем ребро (v, w) в граф 
			adj[v].push_back(w); 
	}
	used.assign(n1, false); mt.assign(n2, -1); 
} 
void solve() { //находим максимальное паросочетание 
	for (int v = 0; v < n1; ++v) { 
		used.assign(n1, false); //если нашли увеличивающую цепь, //то размер максимального паросочетания увеличиваем на 1
		if (kuhn(v)) { 
			mtSize++; 
		} 
	} 
}

void printData() {
	printf("%d\n", mtSize); 
	for (int i = 0; i < n2; ++i) { 
		if (mt[i] != -1) { 
			printf("%d %d\n", mt[i] + 1, i + 1); 
		} 
	} 
} 

int main() { 
	readData(); 
	solve(); 
	printData(); 
	return 0; 
}
