// HNSWDll.cpp : Defines the exported functions for the DLL.
//

#include "pch.h"
#include "framework.h"
#include "HNSWDll.h"


// This is an example of an exported variable
int nHNSWDll = 0;

// This is an example of an exported function.
CHNSWDll* init(void)
{
    data = new CHNSWDll();
    return data;
}

int dispose(void)
{
    delete data;
    return 0;
}

// This is the constructor of a class that has been exported.
CHNSWDll::CHNSWDll()
{
    return;
}
void CHNSWDll::init()
{
    return;
}

Index<float>* Index_Create(const LPCSTR space_name, const int dim, const int debug_mode)
{
    return new Index<float>(space_name, dim, debug_mode);
}

int Index_Delete(Index<float>* index)
{
	delete index;
	return 0;
}

void Print_Info(Index<float>* index, LPCSTR prefix)
{
    std::cout << "[C++ Debug] [" << prefix << "] -"
		<< " dim = " << index->dim
        << " space: " << index->space_name
        << " elements: " << index->appr_alg->cur_element_count
        << "\n";
}

void Index_Init(Index<float>* index, const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed)
{
	index->init_new_index(maxElements, M, efConstruction, random_seed);
}

void Index_Save(Index<FLOAT> *index, LPCSTR path_to_index)
{
    index->saveIndex(path_to_index);
}

void Index_Load(Index<FLOAT>* index, LPCSTR path_to_index, size_t max_elements)
{
    index->loadIndex(path_to_index, max_elements);
}

void Index_AddItems(Index<FLOAT> *index, FLOAT* input, size_t * ids_, int size, int num_threads)
{
    index->addItems(input, ids_, size, num_threads);
}

void Index_Search(Index<FLOAT>* index, FLOAT* query, int qsize, int k, SearchResult* results, int num_threads)
{
    auto internalResults = index->Search(query, qsize, k, num_threads);

    for (int i = 0; i < qsize; ++i)
    {
        results[i].Id = internalResults[i].Id;
        results[i].Size = internalResults[i].Size;

        /* for (int j = 0; j < internalResults[i].Size; ++j)
         {
             results[i].Scores[j] = internalResults[i].Scores[j];
             results[i].Neighbors[j] = internalResults[i].Neighbors[j];
         }*/
    }

    delete internalResults;
}


void Index_Search1(Index<FLOAT>* index, FLOAT* query, int qsize, int k, ItemAndScore* results, size_t* rsizes, int num_threads)
{
    auto internalResults = index->Search(query, qsize, k, num_threads);
    
    for (int i = 0; i < qsize; ++i)
    {
        rsizes[i] = internalResults[i].Size;
        
        for (size_t j = 0; j < internalResults[i].Size; ++j)
         {
        	results[i * k + j].Score = internalResults[i].neighbors[j].Score;
            results[i * k + j].Item = internalResults[i].neighbors[j].Item;
         }
    }

    delete internalResults;
}

