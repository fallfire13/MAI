#include <iostream>

// const int buckets_num = 100001;

class TNote
{
public:
    unsigned long long Key;
    unsigned long long Value;
    TNote() {}
    TNote( unsigned long long key, unsigned long long value){
        Key = key;
        Value = value;
    }

};

void InsertionSort(TNote *arr, int size){
    for (int i = 1; i < size; ++i){
        TNote tmp = arr[i];
        for (int j = i - 1; j >= 0; --j){
            if (tmp.Key >= arr[j].Key){
                break;
            }
            arr[j + 1] = arr[j];
            arr[j] = tmp;
        }

    }
}

void BucketSort(TNote *notes, int size, unsigned long long max) {
    if (max == 0){
        for (size_t i = 0; i < size; ++i){
                std::cout << notes[i].Key << '\t' << notes[i].Value << std::endl;
        }
        return;
    }

    const int buckets_num = size;
    TNote **buckets = new TNote*[buckets_num];
    int *count = new int[buckets_num];
    for (int i = 0; i < buckets_num; ++i){
        count[i] = 0;
    }

    for (size_t i = 0; i < size; ++i){
        int buc_number = ((double)notes[i].Key / (double)max) * (buckets_num-1);
        count[buc_number]++;
    }

    for (int i = 0 ; i < buckets_num; ++i){
        buckets[i] = new TNote[count[i]];
    }

    for (int i = 0 ; i < buckets_num; ++i){
        count[i] = 0;
    }

    for (size_t i = 0; i < size; ++i){
        int buc_number = ((double)notes[i].Key / (double)max) * (buckets_num-1);
        buckets[buc_number][count[buc_number]] = notes[i];
        count[buc_number]++;
    }

    for (int i = 0; i < buckets_num; ++i){
        InsertionSort(buckets[i] , count[i]);
    }

    int c = 0;
    for (int i = 0; i < buckets_num; ++i){
        for (int j = 0; j < count[i]; ++j){
            std::cout << buckets[i][j].Key << '\t' << buckets[i][j].Value << std::endl;
            c++;
        }
        delete [] buckets[i];
    }
    delete [] buckets;
    delete [] count;
}

int main(void) {
    std::cin.tie(0);
    std::ios::sync_with_stdio(false);

    unsigned long long key, max_key = 0;
    unsigned long long value;
    TNote in_tmp;
    size_t cap = 4;
    TNote *notes = new TNote[cap];
    TNote *tmp_notes;
    size_t i = 0;

    while (std::cin >> key >> value) {
        if (key > max_key){
            max_key = key;
        }
        if (i >= cap){
            cap = cap * 3 / 2;
            tmp_notes = new TNote[cap];

            for (int k = 0; k < i; ++k){
                tmp_notes[k] = notes[k];
            }

            delete [] notes;
            notes = tmp_notes;
        }
        notes[i] = TNote(key , value);
        i++;
    }

    cap = i;

    tmp_notes = new TNote[cap];
    for (int k = 0; k < i; ++k){
        tmp_notes[k] = notes[k];
    }
    delete [] notes;
    notes = tmp_notes;

    BucketSort(notes, i , max_key);
    delete [] notes;

    return 0;
}
