// See https://aka.ms/new-console-template for more information

using System.Diagnostics;
using System.Runtime.CompilerServices;
using HNSW;
using Microsoft.Xbox.Recommendations.Modeling.ETS.Instances.Common;
using NumpyIO;

internal class HnswTester
{
    public HnswTester(bool debugMode)
    {
        DebugMode = debugMode;
    }
    public bool DebugMode { get; set; }


    public void Run(int maxDegreeOfParallelism)
    {
        int hnswK = 1000;
        int groundTruthK = 250;
        int? maxDataSize = 100000;

        var mParam = 32;
        var efConstruction = 800;

        var datasetName = "Wines";
        var inputPath = @"C:\Users\saaamar\OneDrive - Microsoft\temp\hnsw-benchmark\datasets\wines";
        var outputPath = @"c:/temp/hnsw/results";

        var reducedDimensionDatasetFileName = Path.Join(inputPath, @"wines120kcosine-128.npz");
        var originalDimensionDatasetFileName = Path.Join(inputPath, @"wines120kcosine-1024.npz");

        PrepareDataset(true, reducedDimensionDatasetFileName, out var embeddedVectorsListReduced, out var textListReduced, maxDataSize);
        PrepareDataset(true, originalDimensionDatasetFileName, out var embeddedVectorsListOriginal, out var textListOriginal, maxDataSize);

        var seedsIndexList = Enumerable.Range(0, 100).ToArray();

        Console.WriteLine($"Search in original dataset dim={embeddedVectorsListOriginal[0].Length}, data size={embeddedVectorsListOriginal.Length}");

        var groundTruthResults = BuildGroundTruthResults(maxDegreeOfParallelism, groundTruthK, datasetName, embeddedVectorsListOriginal, seedsIndexList);

        TestExactScoreAndOrderBySort(maxDegreeOfParallelism, embeddedVectorsListReduced, groundTruthK, datasetName, seedsIndexList, groundTruthResults, hnswK, embeddedVectorsListOriginal);
        TestPriorityQueueScoreAndSort(maxDegreeOfParallelism, embeddedVectorsListOriginal, groundTruthK, datasetName, seedsIndexList, groundTruthResults, embeddedVectorsListReduced, hnswK);
        TestHNSWScoreAndSort(maxDegreeOfParallelism, embeddedVectorsListOriginal, hnswK, datasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults);
        TestHNSWScoreAndSort(maxDegreeOfParallelism, embeddedVectorsListOriginal, groundTruthK, datasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults);
    }

    private (int candidateIndex, float Score)[][] BuildGroundTruthResults(
        int maxDegreeOfParallelism,
        int groundTruthK,
        string datasetName,
        float[][] embeddedVectorsListOriginal,
        int[] seedsIndexList)
    {
        var ss = new ScoreAndSortExact($"GT OrderBySort [k={groundTruthK}]", maxDegreeOfParallelism, groundTruthK, datasetName, embeddedVectorsListOriginal, SIMDCosineSimilarityVectorsScoreForUnits);
        var groundTruthResults = ss.Run(seedsIndexList);
        PrintLog(groundTruthResults);

        return groundTruthResults;
    }

    private void TestHNSWScoreAndSort(int maxDegreeOfParallelism, float[][] embeddedVectorsListOriginal, int k,
        string datasetName, string outputPath, int mParam, int efConstruction, int[] seedsIndexList,
        (int candidateIndex, float Score)[][] groundTruthResults)
    {
        {
            using (var hnsw = new ScoreAndSortHNSWCpp($"HNSW-C++ [k={k}]", maxDegreeOfParallelism, k, datasetName, embeddedVectorsListOriginal, SIMDCosineDistanceVectorsScoreForUnits))
            {
                hnsw.Init(outputPath, embeddedVectorsListOriginal.Length, mParam, efConstruction);
                var resultsHNSW = hnsw.Run(seedsIndexList, groundTruthResults);
                PrintLog(resultsHNSW);
            }
        }

        {
            var hnsw = new ScoreAndSortHNSW($"HNSW-C# [k={k}]", maxDegreeOfParallelism, k, datasetName, embeddedVectorsListOriginal, SIMDCosineDistanceVectorsScoreForUnits);
            hnsw.Init(outputPath, mParam, efConstruction);
            var resultsHNSW = hnsw.Run(seedsIndexList, groundTruthResults);
            PrintLog(resultsHNSW);
        }
    }

    private void TestPriorityQueueScoreAndSort(
        int maxDegreeOfParallelism,
        float[][] embeddedVectorsListOriginal,
        int groundTruthK,
        string datasetName,
        int[] seedsIndexList,
        (int candidateIndex, float Score)[][] groundTruthResults,
        float[][] embeddedVectorsListReduced,
        int hnswK)
    {
        {
            var heap = new ScoreAndSortHeapSort($"Pr. Queue [k={groundTruthK}]", maxDegreeOfParallelism, groundTruthK, datasetName, embeddedVectorsListOriginal, SIMDCosineSimilarityVectorsScoreForUnits);
            var resultsHeap = heap.Run(seedsIndexList, groundTruthResults);
            PrintLog(resultsHeap);
        }

        {
            var heap = new ScoreAndSortHeapSort($"Pr.Queue [k={groundTruthK}]", maxDegreeOfParallelism, groundTruthK, datasetName, embeddedVectorsListReduced, SIMDCosineSimilarityVectorsScoreForUnits);
            var resultsHeap = heap.Run(seedsIndexList, groundTruthResults);
            PrintLog(resultsHeap);
        }

        {
            var heap = new ScoreAndSortHeapSort($"Pr.Queue [k={groundTruthK}/{hnswK}]", maxDegreeOfParallelism, hnswK, datasetName, embeddedVectorsListReduced, SIMDCosineSimilarityVectorsScoreForUnits, embeddedVectorsListOriginal, groundTruthK);
            var resultsHeap = heap.Run(seedsIndexList, groundTruthResults);
            PrintLog(resultsHeap);
        }
    }

    private void TestExactScoreAndOrderBySort(int maxDegreeOfParallelism, float[][] embeddedVectorsListReduced,
        int groundTruthK,
        string datasetName, int[] seedsIndexList, (int candidateIndex, float Score)[][] groundTruthResults, int hnswK,
        float[][] embeddedVectorsListOriginal)
    {
        {
            var exactSearchReduced = new ScoreAndSortExact($"OrderBySort [k={groundTruthK}]", maxDegreeOfParallelism, groundTruthK, datasetName, embeddedVectorsListReduced, SIMDCosineSimilarityVectorsScoreForUnits);
            var resultsExactReduced = exactSearchReduced.Run(seedsIndexList, groundTruthResults);
            PrintLog(resultsExactReduced);
        }
        
        {
            var exactSearchReduced = new ScoreAndSortExact($"OrderBySort [k={groundTruthK}/{hnswK}]", maxDegreeOfParallelism, hnswK, datasetName, embeddedVectorsListReduced, SIMDCosineSimilarityVectorsScoreForUnits, embeddedVectorsListOriginal, groundTruthK);
            var resultsExactReduced = exactSearchReduced.Run(seedsIndexList, groundTruthResults);
            PrintLog(resultsExactReduced);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float SIMDCosineSimilarityVectorsScoreForUnits(float[] l, float[] r)
    {
        return VectorUtilities.SIMDCosineSimilarityVectorsScoreForUnits(l, r);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float SIMDCosineDistanceVectorsScoreForUnits(float[] l, float[] r)
    {
        return 1F - VectorUtilities.SIMDCosineSimilarityVectorsScoreForUnits(l, r);
    }

    private void PrintLog((int candidateIndex, float Score)[][] results)
    {
        if(!DebugMode)
            return;
        Console.WriteLine($"number of seeds {results.Length}");

        if (results.Length == 0)
            return;

        for (var i = 0; i < results[0].Length; i++)
        {
            var reco = results[0][i];
            Console.WriteLine($"\t{reco.candidateIndex} , {reco.Score}");
            if (i == 10)
                break;
        }
    }

    private static void PrepareDataset(
        bool doNormalizeFunction,
        string fullFileName,
        out float[][] embeddedVectorsList,
        out List<string> textList,
        int? maxDataSize = null)
    {
        textList = new List<string>();
        
        //if (fullFileName.EndsWith("npy"))
        //{
        //    ReadNpyVectorsFromFile(inputPath, out embeddedVectorsList);
        //}
        //else
        if (fullFileName.EndsWith("npz"))
        {
            ReadNpzVectorsFromFile(fullFileName, out embeddedVectorsList, maxDataSize);
        }
        else if (fullFileName.EndsWith("csv"))
        {
            ReadRawVectorsFromFile(fullFileName, out embeddedVectorsList, maxDataSize);
        }
        else
        {
            throw new Exception("Unsupported extension");
        }

        //ReadRawTextFromFile(fullFileName, candidatesAll.Count, out textAll);
        
        if (doNormalizeFunction)
        {
            Console.WriteLine($"Running normalization for {embeddedVectorsList.Length} items.");
            embeddedVectorsList.AsParallel().WithDegreeOfParallelism(Environment.ProcessorCount).ForAll(VectorUtilities.SIMDNormalize);
        }
    }


    //private static void ReadNpyVectorsFromFile(string pathPrefix, out List<float[]> candidates, int dataSizeLimit)
    //{
    //    candidates = new List<float[]>();
    //    var noFileFound = true;

    //    foreach (var i in Enumerable.Range(-1, 10))
    //    {
    //        var npyFilename = @$"{pathPrefix}" + (i == -1 ? "" : @$".{i}") + ".npy";
    //        Console.WriteLine(npyFilename);

    //        if (!File.Exists(npyFilename))
    //            continue;

    //        noFileFound = false;

    //        var v = np.load(npyFilename); //NDArray

    //        var tempList = v.astype(np.float32)
    //            .ToJaggedArray<float>()
    //            .OfType<float[]>()
    //            .Take(dataSizeLimit)
    //            .Select(a => { return a.OfType<float>().ToArray(); })
    //            .ToList();

    //        candidates.AddRange(tempList);

    //        if (candidates.Count >= dataSizeLimit)
    //        {
    //            break;
    //        }
    //    }

    //    if (noFileFound)
    //        throw new FileNotFoundException($"No relevant input file found {pathPrefix}");
    //}

    private static void ReadNpzVectorsFromFile(string fullFileName, out float[][] candidates, int? dataSizeLimit = null)
    {
        NPZInputStream npz = new NPZInputStream(fullFileName);
        var keys = npz.Keys();
        var header = npz.Peek(keys[0]);

        var shape = header.Shape;
        IList<float> values;

        if (header.DataType == DataType.FLOAT32)
        {
            var dataAsFloat32 = npz.ReadFloat32(keys[0]);
            values = dataAsFloat32.Values;
        }
        else if (header.DataType == DataType.FLOAT64)
        {
            var dataAsFloat64 = npz.ReadFloat64(keys[0]);
            values = dataAsFloat64.Values.Select(a => (float)a).ToArray();
        }
        else
        {
            throw new NotImplementedException("not supported");
        }

        int dataSize = ((int)shape[0]);
        int vsize = ((int)shape[1]);

        if (dataSizeLimit.HasValue && dataSize > dataSizeLimit)
        {
            dataSize = dataSizeLimit.Value;
        }

        candidates = Enumerable.Range(0, dataSize).Select(i =>
        {
            float[] val = values.Skip(i * vsize).Take(vsize).ToArray();
            return val;
        }).ToArray();

        return;

        //var fb = new Float32Buffer(t.Values);
        //candidates = Enumerable.Range(0, ((int)shape[0])).Select(i =>
        //{
        //    int vsize = ((int) shape[1]);
        //    float[] val = new float[vsize];
        //    fb.CopyTo(i * vsize, val, 0, vsize);
        //    return val;
        //}).ToList();

        //var singleContent = np.Load_Npz<float[]>(npyFilename);
        //var data1 = singleContent.Values.Take(10).Select(a => a.Length).ToArray();
        //var data = singleContent["data.npy"];
        //var singleNDArray = np.array(data); // type is NDArray
        //var singleArray = singleNDArray.ToArray<float>(); // type is float[]

        //candidates = Enumerable.Empty<float[]>().ToList();
        //candidates = v
        //    .astype(np.float32)
        //    .ToJaggedArray<float>()
        //    .OfType<float[]>()
        //    .Select(a =>
        //    {
        //        return a.OfType<float>().ToArray();
        //    })
        //    .ToList();
    }

    private static void ReadRawVectorsFromFile(string fullFileName, out float[][] candidates, int? dataSizeLimit = null)
    {
        var clock = Stopwatch.StartNew();

        var vectorsFileName = fullFileName + "_vectors.csv";

        Console.Write($"Read vectors from file {vectorsFileName} ...");

        //var vectors = new float[titleDescription.Count][];
        //Parallel.ForEach(
        //    File.ReadLines(vectorsFileName),
        //    new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount },
        //    (line, state, i) =>
        //    {
        //        vectors[i] = line.Split(",").Select(float.Parse).ToArray();
        //    }
        //);

        //candidates = vectors.ToList();

        Console.Write($"Building object ...");
        var lines = File.ReadAllLines(vectorsFileName);


        IEnumerable<string> tempEnumerable;
        if (dataSizeLimit.HasValue && lines.Length > dataSizeLimit)
        {
            tempEnumerable = lines.Take(dataSizeLimit.Value);
        }
        else
        {
            tempEnumerable = lines;
        }

        candidates = tempEnumerable.AsParallel()
            .AsOrdered()
            .WithDegreeOfParallelism(Environment.ProcessorCount)
            .Select(line => line.Split(",").Select(float.Parse).ToArray())
            .ToArray();

        clock.Stop();
        Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");

        //CollectRunTime($"Load items {candidates.Count}", clock.ElapsedMilliseconds.ToString());
    }


}