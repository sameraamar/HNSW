// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the HNSWDLL_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// HNSWDLL_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
//#ifdef HNSWDLL_EXPORTS
//#define HNSWDLL_API __declspec(dllexport)
//#else
//#define HNSWDLL_API __declspec(dllimport)
//#endif


#pragma once

#ifdef MATHLIBRARY_EXPORTS
#define HNSWDLL_API __declspec(dllexport)
#else
#define HNSWDLL_API __declspec(dllimport)
#endif

#include "hnswlib/hnswlib.h"

// This class is exported from the dll
extern "C" class HNSWDLL_API CHNSWDll {
public:
	CHNSWDll(void);
	// TODO: add your methods here.

	void init();
};

CHNSWDll* data;

extern "C" HNSWDLL_API int nHNSWDll;

extern "C" HNSWDLL_API CHNSWDll * init(void);
extern "C" HNSWDLL_API int dispose(void);

struct ItemAndScore
{
	hnswlib::labeltype Item;
    float Score;
};

struct SearchResult
{
	hnswlib::labeltype Id;
    size_t Size;
    ItemAndScore* neighbors;
};

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    }
    else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if ((id >= end)) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    }
                    catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
                }));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }


}



template<typename dist_t, typename data_t = float>
class Index {
public:
    Index(const std::string& space_name, const int dim, const int debugMode) :
        space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            l2space = new hnswlib::L2Space(dim);
        }
        else if (space_name == "ip") {
            l2space = new hnswlib::InnerProductSpace(dim);
        }
        else if (space_name == "cosine") {
            l2space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        }
        else {
            throw new std::runtime_error("Space name must be one of l2, ip, or cosine.");
        }
        appr_alg = NULL;
        ep_added = true;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();

        default_ef = 10;
        debug_mode = debugMode;
    }

    static const int ser_version = 1; // serialization version

    std::string space_name;
    int debug_mode;
    int dim;
    size_t seed;
    size_t default_ef;

    bool index_inited;
    bool ep_added;
    bool normalize;
    int num_threads_default;
    hnswlib::labeltype cur_l;
    hnswlib::HierarchicalNSW<dist_t>* appr_alg;
    hnswlib::SpaceInterface<FLOAT>* l2space;

    ~Index() {
        delete l2space;
        if (appr_alg)
            delete appr_alg;

        if(debug_mode)
	        std::cout << "[C++]: ~Index() is called\n";
    }

    void init_new_index(const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed) {
        if (appr_alg) {
            throw new std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, maxElements, M, efConstruction, random_seed);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        seed = random_seed;
    }

    void saveIndex(const std::string& path_to_index) {
        //appr_alg->checkIntegrity();
        appr_alg->saveIndex(path_to_index);
    }

    void loadIndex(const std::string& path_to_index, size_t max_elements) {
        if (appr_alg) {
            std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated.";
            delete appr_alg;
        }
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, path_to_index, false, max_elements);
        cur_l = appr_alg->cur_element_count;
        index_inited = true;

        if (debug_mode)
	        std::cout << "M=" << appr_alg->M_
	            << ", data size=" << appr_alg->data_size_
	            << ", cur_element_count=" << appr_alg->cur_element_count
	            << ", element_levels_.size=" << appr_alg->element_levels_.size()
	            << "\n";

        //appr_alg->checkIntegrity();
    }

    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;

        //std::cout << "[C++][Debug] normalizing norm = " << norm << ", " << norm_array[0] << "\n";
    }

    void addItems(FLOAT* input, size_t* ids_, int rows, int num_threads = -1)
    {
        if (num_threads <= 0)
            num_threads = num_threads_default;

        if (debug_mode)
            std::cout << "Adding " << rows << " elements with dimension " << dim;

        std::vector<size_t> ids(rows);
        for (size_t i = 0; i < ids.size(); i++) {
            ids[i] = ids_[i];
        }
        
        /*
        if (features != dim)
            throw std::runtime_error("wrong dimensionality of the vectors");
        */

        // avoid using threads when the number of searches is small:
        if (rows <= num_threads * 4) {
            num_threads = 1;
        }

        //if (!ids_.is_none()) {
        //    py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
        //    auto ids_numpy = items.request();
        //    if (ids_numpy.ndim == 1 && ids_numpy.shape[0] == rows) {
        //        std::vector<size_t> ids1(ids_numpy.shape[0]);
        //        for (size_t i = 0; i < ids1.size(); i++) {
        //            ids1[i] = items.data()[i];
        //        }
        //        ids.swap(ids1);
        //    }
        //    else if (ids_numpy.ndim == 0 && rows == 1) {
        //        ids.push_back(*items.data());
        //    }
        //    else
        //        throw std::runtime_error("wrong dimensionality of the labels");
        //}


        {

            int start = 0;
            if (!ep_added) {
                size_t id = ids.size() ? ids.at(0) : (cur_l);
                float* vector_data = input;
                std::vector<FLOAT> norm_array(dim);
                if (normalize) {
                    normalize_vector(vector_data, norm_array.data());
                    vector_data = norm_array.data();
                }
                appr_alg->addPoint((void*)vector_data, (size_t)id);
                start = 1;
                ep_added = true;
            }

            //py::gil_scoped_release l;
            if (normalize == false) {
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void*)(input + row * dim), (size_t)id);
                    });
            }
            else {
                std::vector<FLOAT> norm_array(num_threads * dim);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    // normalize vector:
                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)(input + row * dim), (norm_array.data() + start_idx));

                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void*)(norm_array.data() + start_idx), (size_t)id);
                    });
            };
            cur_l += rows;
        }

        if (debug_mode)            
            std::cout << ". Elements = " << appr_alg->cur_element_count << "\n";
    }


    std::vector<std::vector<data_t>> getDataReturnList(size_t* ids_, int rows) {

        std::vector<size_t> ids(rows);
        for (size_t i = 0; i < ids.size(); i++) {
            ids[i] = ids_[i];
        }

        /*  if (!ids_.is_none()) {
              py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
              auto ids_numpy = items.request();
              std::vector<size_t> ids1(ids_numpy.shape[0]);
              for (size_t i = 0; i < ids1.size(); i++) {
                  ids1[i] = items.data()[i];
              }
              ids.swap(ids1);
          }*/

        std::vector<std::vector<data_t>> data;
        for (auto id : ids) {
            data.push_back(appr_alg->template getDataByLabel<data_t>(id));
        }
        return data;
    }

    std::vector<hnswlib::labeltype> getIdsList() {

        std::vector<hnswlib::labeltype> ids;

        for (auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }


    SearchResult* Search(FLOAT* input, size_t rows, size_t k = 1, int num_threads = -1) {

        hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        const size_t features = dim;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            // avoid using threads when the number of searches is small:

            if (rows <= num_threads * 4) {
                num_threads = 1;
            }

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];

            if (normalize == false) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)(input + row), k);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contigious 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                    }
                );
            }
            else {
                std::vector<float> norm_array(num_threads * features);

                //std::cout << "C++ Debug - 1 - rows = " << rows << "\n";

                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t start_idx = threadId * dim;
                    normalize_vector(input + row, (norm_array.data() + start_idx));
                    
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)(norm_array.data() + start_idx), k);

                    //std::cout << "C++ Debug - 3.1 - " << row << " results: " << result.size() << "\n";
                    
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contigious 2D array. Probably ef or M is too small");
                    
                    for (int i = k - 1; i >= 0; i--) {
                        //std::cout << "C++ Debug inside for - " << i << "\n";

                    	auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;

                        result.pop();
                    }

                    //std::cout << "C++ Debug - 3.2 - end of ParallelFor " << row << "\n";

                    }
                );
            }

        }
        //std::cout << "C++ Debug - 4 - finished... \n";

        SearchResult* results = new SearchResult[rows];
        ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
            //std::cout << "C++ Debug - 5 - copy results ... " << " results: " << row << "\n";

            results[row].Id = row;
            results[row].Size = k;
            
            results[row].neighbors = new ItemAndScore[k];

            for (int i = k - 1; i >= 0; i--)
            {
                results[row].neighbors[i].Item = data_numpy_l[row * k + i];
                results[row].neighbors[i].Score = data_numpy_d[row * k + i];
            }
            }
        );

        return results;
        //return py::make_tuple(
        //    py::array_t<hnswlib::labeltype>(
        //        { rows, k }, // shape
        //        { k * sizeof(hnswlib::labeltype),
        //         sizeof(hnswlib::labeltype) }, // C-style contiguous strides for double
        //        data_numpy_l, // the data pointer
        //        free_when_done_l),
        //    py::array_t<dist_t>(
        //        { rows, k }, // shape
        //        { k * sizeof(dist_t), sizeof(dist_t) }, // C-style contiguous strides for double
        //        data_numpy_d, // the data pointer
        //        free_when_done_d));
    }

};
 
struct Item
{
    FLOAT* data;
    int dim;
};

extern "C" HNSWDLL_API Index<FLOAT> *Index_Create(const LPCSTR space_name, const int dim, const int debug_mode);
extern "C" HNSWDLL_API int Index_Delete(Index<FLOAT> *index);
extern "C" HNSWDLL_API void Index_Init(Index<FLOAT> *index, const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed);
extern "C" HNSWDLL_API void Print_Info(Index<float>*index, LPCSTR prefix);
extern "C" HNSWDLL_API void Index_Save(Index<FLOAT> *index, LPCSTR path_to_index);
extern "C" HNSWDLL_API void Index_Load(Index<FLOAT> *index, LPCSTR path_to_index, size_t max_elements);
extern "C" HNSWDLL_API void Index_AddItems(Index<FLOAT> *index, FLOAT * input, size_t * ids_, int size, int num_threads = -1);
extern "C" HNSWDLL_API void Index_Search(Index<FLOAT> *index, FLOAT * query, int qsize, int k, SearchResult * results, int num_threads = -1);
extern "C" HNSWDLL_API void Index_Search1  (Index<FLOAT> *index, FLOAT * query, int qsize, int k, ItemAndScore * results, size_t * rsizes, int num_threads = -1);

