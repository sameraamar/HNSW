// See https://aka.ms/new-console-template for more information

using System.Collections;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using HNSW;
using Microsoft.Xbox.Recommendations.Modeling.ETS.Instances.Common;
using NumpyIO;

internal class HnswTester
{
    private string datasetName = "Wines";
    private string inputPath = @"C:\Users\saaamar\OneDrive - Microsoft\temp\hnsw-benchmark\datasets\wines";
    private string outputPath = @"c:/temp/hnsw/results";

    private float[][] _embeddedVectorsListOriginal;
    private float[][] _embeddedVectorsListReduced;
    private List<string> _textListOriginal;
    private List<string> _textListReduced;

    //
    //private (int candidateIndex, float Score)[][] _groundTruthResults;
    //private TimeSpan _groundTruthElapsedTime;

    public HnswTester(bool debugMode)
    {
        DebugMode = debugMode;
    }
    public bool DebugMode { get; set; }

    public void Run(int maxDegreeOfParallelism)
    {
        int maxDataSize = 100000;
        var mParam = 32;
        var efConstruction = 800;

        int groundTruthK = 250;

        LoadDatabase(maxDataSize, maxDegreeOfParallelism, groundTruthK);
        var seedsIndexList = Enumerable.Range(0, 100).ToArray();

        var dataSizes = Enumerable
            .Range(1, 10)
            .Select(a => a * 10000)
            .ToArray();

        (int candidateIndex, float Score)[][] groundTruthResults;  
        TimeSpan groundTruthRuntime;

        foreach (var size in dataSizes)
        {
            var partialDataArray = _embeddedVectorsListOriginal.Take(size).ToArray();
            BuildGroundTruthResults(maxDegreeOfParallelism, groundTruthK, datasetName, partialDataArray, seedsIndexList, out groundTruthResults, out groundTruthRuntime);
            TestHNSWScoreAndSortCpp(0.95f, maxDegreeOfParallelism, partialDataArray.Take(size), datasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);
        }


        BuildGroundTruthResults(maxDegreeOfParallelism, groundTruthK, datasetName, _embeddedVectorsListOriginal, seedsIndexList, out groundTruthResults, out groundTruthRuntime);

        TestExactScoreAndOrderBySort(maxDegreeOfParallelism, _embeddedVectorsListReduced, groundTruthK, datasetName, seedsIndexList, groundTruthResults, groundTruthRuntime);
        TestExactScoreAndOrderBySortExtendedK(maxDegreeOfParallelism, _embeddedVectorsListReduced, groundTruthK, datasetName, seedsIndexList, _embeddedVectorsListOriginal, groundTruthResults, groundTruthRuntime, 0.95f);

        TestPriorityQueueScoreAndSort(maxDegreeOfParallelism, _embeddedVectorsListOriginal, groundTruthK, datasetName, seedsIndexList, groundTruthResults, groundTruthRuntime);
        TestPriorityQueueScoreAndSort(maxDegreeOfParallelism, _embeddedVectorsListReduced, groundTruthK, datasetName, seedsIndexList, groundTruthResults, groundTruthRuntime);


        TestPriorityQueueScoreAndSortExtendedK(maxDegreeOfParallelism, _embeddedVectorsListOriginal, groundTruthK, datasetName, seedsIndexList, _embeddedVectorsListReduced, groundTruthResults, groundTruthRuntime, 0.95f);

        TestHNSWScoreAndSortCSharp(maxDegreeOfParallelism, _embeddedVectorsListOriginal, groundTruthK, datasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);
        TestHNSWScoreAndSortCpp(maxDegreeOfParallelism, _embeddedVectorsListOriginal, groundTruthK, datasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);

        foreach (var size in dataSizes)
        {
            var partialDataArray = _embeddedVectorsListReduced.Take(size).ToArray();
            BuildGroundTruthResults(maxDegreeOfParallelism, groundTruthK, datasetName, partialDataArray, seedsIndexList, out groundTruthResults, out groundTruthRuntime);
            TestExactScoreAndOrderBySortExtendedK(maxDegreeOfParallelism, partialDataArray, groundTruthK, datasetName, seedsIndexList, _embeddedVectorsListOriginal, groundTruthResults, groundTruthRuntime, 0.95f);
        }

    }

    public void LoadDatabase(int maxDataSize, int maxDegreeOfParallelism, int groundTruthK)
    {
        var reducedDimensionDatasetFileName = Path.Join(inputPath, @"wines120kcosine-128.npz");
        var originalDimensionDatasetFileName = Path.Join(inputPath, @"wines120kcosine-1024.npz");

        PrepareDataset(true, reducedDimensionDatasetFileName, out _embeddedVectorsListReduced, out _textListReduced, maxDataSize);
        PrepareDataset(true, originalDimensionDatasetFileName, out _embeddedVectorsListOriginal, out _textListOriginal, maxDataSize);
    }

    private void BuildGroundTruthResults(
        int maxDegreeOfParallelism,
        int groundTruthK,
        string datasetName,
        float[][] embeddedVectorsListOriginal,
        int[] seedsIndexList,
        out (int candidateIndex, float Score)[][] groundTruthResults,
        out TimeSpan groundTruthElapsedTime)
    {
        var ss = new ScoreAndSortExact($"GT OrderBySort [k={groundTruthK}]", maxDegreeOfParallelism, groundTruthK, datasetName, embeddedVectorsListOriginal, SIMDCosineSimilarityVectorsScoreForUnits);
        ss.Run(seedsIndexList);
        PrintDataSampleDebug(ss.Results);

        groundTruthResults = ss.Results;
        groundTruthElapsedTime = ss.ElapsedTime;

        ss.Evaluate(seedsIndexList, groundTruthResults, groundTruthElapsedTime);
    }

    private void TestHNSWScoreAndSortCpp(int maxDegreeOfParallelism, IEnumerable<float[]> embeddedVectorsList, int k, string datasetName, string outputPath, int mParam, int efConstruction, int[] seedsIndexList,
        (int candidateIndex, float Score)[][] groundTruthResults,
        TimeSpan groundTruthElapsedTime)
    {
        var inputDataList = embeddedVectorsList.ToArray();
        using (var hnsw = new ScoreAndSortHNSWCpp($"HNSW-C++ [k={k}]", maxDegreeOfParallelism, k, datasetName, inputDataList, SIMDCosineDistanceVectorsScoreForUnits, DebugMode))
        {
            hnsw.Init(outputPath, inputDataList.Length, mParam, efConstruction);
            hnsw.Run(seedsIndexList);
            hnsw.Evaluate(seedsIndexList, groundTruthResults, groundTruthElapsedTime);
            PrintDataSampleDebug(hnsw.Results);
        }
    }

    private void TestHNSWScoreAndSortCpp(float desiredRecall, int maxDegreeOfParallelism, IEnumerable<float[]> embeddedVectorsList, string datasetName, string outputPath, int mParam, int efConstruction, int[] seedsIndexList,
        (int candidateIndex, float Score)[][] groundTruthResults,
        TimeSpan groundTruthElapsedTime)
    {
        var inputDataList = embeddedVectorsList.ToArray();

        foreach (var i in Enumerable.Range(1, 100))
        {
            var k = i * 10;
            using (var hnsw = new ScoreAndSortHNSWCpp($"HNSW-C++ [k={k}]", maxDegreeOfParallelism, k, datasetName,
                       inputDataList, SIMDCosineDistanceVectorsScoreForUnits, DebugMode))
            {
                hnsw.Init(outputPath, inputDataList.Length, mParam, efConstruction);
                hnsw.Run(seedsIndexList);
                if (hnsw.EvaluateScoring(groundTruthResults) > desiredRecall)
                {
                    hnsw.Evaluate(seedsIndexList, groundTruthResults, groundTruthElapsedTime);
                    PrintDataSampleDebug(hnsw.Results);
                    break;
                }
            }
        }
    }

    private void TestHNSWScoreAndSortCSharp(int maxDegreeOfParallelism, float[][] embeddedVectorsListOriginal, int k,
        string datasetName, string outputPath, int mParam, int efConstruction, int[] seedsIndexList,
        (int candidateIndex, float Score)[][] groundTruthResults, TimeSpan groundTruthElapsedTime)
    {
        var hnsw = new ScoreAndSortHNSW($"HNSW-C# [k={k}]", maxDegreeOfParallelism, k, datasetName, embeddedVectorsListOriginal, SIMDCosineDistanceVectorsScoreForUnits);
        hnsw.Init(outputPath, mParam, efConstruction);
        hnsw.Run(seedsIndexList);
        hnsw.Evaluate(seedsIndexList, groundTruthResults, groundTruthElapsedTime);
        PrintDataSampleDebug(hnsw.Results);
    }

    private void TestPriorityQueueScoreAndSort(
        int maxDegreeOfParallelism,
        float[][] embeddedVectorsList,
        int k,
        string datasetName,
        int[] seedsIndexList,
        (int candidateIndex, float Score)[][] groundTruthResults, 
        TimeSpan groundTruthElapsedTime)
    {
        var heap = new ScoreAndSortHeapSort($"Pr. Queue [k={k}]", maxDegreeOfParallelism, k, datasetName, embeddedVectorsList, SIMDCosineSimilarityVectorsScoreForUnits);
        heap.Run(seedsIndexList);
        heap.Evaluate(seedsIndexList, groundTruthResults, groundTruthElapsedTime);
        PrintDataSampleDebug(heap.Results);
    }

    private void TestPriorityQueueScoreAndSortExtendedK(
        int maxDegreeOfParallelism,
        float[][] embeddedVectorsListOriginal,
        int k,
        string datasetName,
        int[] seedsIndexList,
        float[][] embeddedVectorsListReduced,
        (int candidateIndex, float Score)[][] groundTruthResults,
        TimeSpan groundTruthElapsedTime,
        float desiredRecall)
    {
        var heap = new ScoreAndSortHeapSort($"Pr.Queue [k={k}/{extendedK}]", maxDegreeOfParallelism, extendedK, datasetName, embeddedVectorsListReduced, SIMDCosineSimilarityVectorsScoreForUnits, embeddedVectorsListOriginal, k);
        heap.Run(seedsIndexList);
        heap.Evaluate(seedsIndexList, groundTruthResults, groundTruthElapsedTime);
        PrintDataSampleDebug(heap.Results);
    }

    private void TestExactScoreAndOrderBySort(int maxDegreeOfParallelism, float[][] embeddedVectorsListReduced,
        int groundTruthK,
        string datasetName, int[] seedsIndexList, (int candidateIndex, float Score)[][] groundTruthResults,
        TimeSpan groundTruthElapsedTime)
    {
        var exactSearchReduced = new ScoreAndSortExact($"OrderBySort [k={groundTruthK}]", maxDegreeOfParallelism, groundTruthK, datasetName, embeddedVectorsListReduced, SIMDCosineSimilarityVectorsScoreForUnits);
        exactSearchReduced.Run(seedsIndexList);
        exactSearchReduced.Evaluate(seedsIndexList, groundTruthResults, groundTruthElapsedTime);
        PrintDataSampleDebug(exactSearchReduced.Results);
    }

    private void TestExactScoreAndOrderBySortExtendedK(int maxDegreeOfParallelism, float[][] embeddedVectorsListReduced,
        int groundTruthK,
        string datasetName, int[] seedsIndexList,
        float[][] embeddedVectorsListOriginal, 
        (int candidateIndex, float Score)[][] groundTruthResults,
        TimeSpan groundTruthElapsedTime,
        float desiredRecall)
    {
        for (var i = 1; i <= embeddedVectorsListOriginal.Length / 10; i++)
        {
            var extendedK = i * 10;

            var exactSearchReduced = new ScoreAndSortExact($"OrderBySort [k={groundTruthK}/{extendedK}]", maxDegreeOfParallelism, extendedK, datasetName, embeddedVectorsListReduced, SIMDCosineSimilarityVectorsScoreForUnits, embeddedVectorsListOriginal, groundTruthK);
            exactSearchReduced.Run(seedsIndexList);
            if (exactSearchReduced.EvaluateScoring(groundTruthResults) > desiredRecall)
            {
                exactSearchReduced.Evaluate(seedsIndexList, groundTruthResults, groundTruthElapsedTime);
                PrintDataSampleDebug(exactSearchReduced.Results);
                break;
            }
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

    private void PrintDataSampleDebug((int candidateIndex, float Score)[][] results)
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
        using NPZInputStream npz = new NPZInputStream(fullFileName);
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