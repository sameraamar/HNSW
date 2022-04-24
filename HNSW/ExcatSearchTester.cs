// See https://aka.ms/new-console-template for more information

using HNSW;
using Microsoft.Xbox.Recommendations.Modeling.ETS.Instances.Common;
using NumpyIO;
using System.Diagnostics;

internal class ExactSearchTester : BaseTester
{  
    private string inputPath;

    private string _inputOriginalDataFileName;

    public float[][] _embeddedVectorsListOriginal;
    public List<string> _textListOriginal;

    public int[] seedsIndexList;

    public int GroundTruthK { get; }
    public int SeedsCount { get; }

    public bool UseHeapSort { get; }
    
    public (int candidateIndex, float Score)[][] groundTruthResults { get; private set; }
    public TimeSpan groundTruthRuntime { get; private set; }

    public ExactSearchTester(string datasetName, string inputPath, string inputOriginalDataFileName, bool debugMode, bool useHeapSort, int groundTruthK, int seedsCount)
        : base(datasetName, debugMode)
    {
        this.inputPath = inputPath;
        this._inputOriginalDataFileName = inputOriginalDataFileName;
        UseHeapSort = useHeapSort;
        GroundTruthK = groundTruthK;
        SeedsCount = seedsCount;
    }


    public void Run(int maxDegreeOfParallelism, int? maxDataSize = null)
    {
        LoadDatabase(maxDataSize);
        seedsIndexList = Enumerable.Range(0, SeedsCount).ToArray();
        
        BuildGroundTruthResults(maxDegreeOfParallelism, GroundTruthK, DatasetName, _embeddedVectorsListOriginal, seedsIndexList, out var groundTruthResultsTemp, out var groundTruthRuntimeTemp, UseHeapSort);
        groundTruthResults = groundTruthResultsTemp;
        groundTruthRuntime = groundTruthRuntimeTemp;

    }

    public void LoadDatabase(int? maxDataSize)
    {
        var originalDimensionDatasetFileName = Path.Join(inputPath, _inputOriginalDataFileName);
        PrepareDataset(true, originalDimensionDatasetFileName, out _embeddedVectorsListOriginal, out _textListOriginal, maxDataSize);
    }

    public static void PrepareDataset(
        bool doNormalizeFunction,
        string fullFileName,
        out float[][] embeddedVectorsList,
        out List<string> textList,
        int? maxDataSize = null)
    {
        textList = new List<string>();

        //if (fullFileName.EndsWith("npy"))
        //{
        //    ReadNpyVectorsFromFile(fullFileName, out embeddedVectorsList, maxDataSize);
        //}
        //else
        if (fullFileName.EndsWith("npz"))
        {
            ReadNpzVectorsFromFile(fullFileName, out embeddedVectorsList, maxDataSize);
        }
        else if (fullFileName.EndsWith("csv"))
        {
            ReadRawCsvFromFile(fullFileName, out embeddedVectorsList, maxDataSize);

            // ReadRawVectorsFromFile(fullFileName, out embeddedVectorsList, maxDataSize);
        }
        else
        {
            throw new Exception("Unsupported extension");
        }


        if (doNormalizeFunction)
        {
            embeddedVectorsList.AsParallel().WithDegreeOfParallelism(Environment.ProcessorCount).ForAll(VectorUtilities.SIMDNormalize);
        }
    }


    private static void ReadRawCsvFromFile(string fullFileName, out float[][] candidates, int? dataSizeLimit = null)
    {
        int headerRows = 0;
        Console.Write($"ReadRawCsvFromFile: Loading file {fullFileName} ... ");

        var candaidatEnumerable = File.ReadLines(fullFileName)
            .Skip(headerRows)
            .AsParallel()
            .AsOrdered()
            .WithDegreeOfParallelism(Environment.ProcessorCount)
            .Select(line => line.Trim())
            .Select(line => line.Split(' '))
            .Select(values => values.Select(a => float.Parse(a)).ToArray());
        if (dataSizeLimit.HasValue)
        {
            candaidatEnumerable = candaidatEnumerable.Take(dataSizeLimit.Value);
        }

        candidates = candaidatEnumerable.ToArray();

        Console.WriteLine($"Done (datasize={candidates.Length}, dim={candidates[0].Length}).");

        //var tempList = new List<float[]>();
        //foreach (var row in csv.Skip(headerRows)
        //    .TakeWhile(r => r.Length > 1 && r.Last().Trim().Length > 0))
        //{
        //    String zerothColumnValue = row[0]; // leftmost column
        //    var firstColumnValue = row[1];
        //}

        //candidates = tempList.Where(a => a != null).ToArray();
    }

    //private static void ReadNpyVectorsFromFile(string pathPrefix, out float[][] candidates, int dataSizeLimit)
    //{
    //    candidates = new float[][]();
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
        Console.Write($"ReadNpzVectorsFromFile: Loading file {fullFileName} ... ");
        Console.Out.Flush();

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

        Console.WriteLine($"Done (datasize={candidates.Length}, dim={candidates[0].Length}).");
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
        Console.Out.Flush();

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

internal class ReducedDimensionHnswTester : BaseTester
{
    private string inputPath;

    private string _inputReducedDataFileName;

    private float[][] _embeddedVectorsListOriginal => gtTester._embeddedVectorsListOriginal;
    public float[][] _embeddedVectorsListReduced;
    public List<string> _textListReduced;

    //private List<string> _textListOriginal;
    //private List<string> _textListReduced;      

    private int groundTruthK => gtTester.GroundTruthK;
    //private const int seedsCount = 100;
    private int[] seedsIndexList => gtTester.seedsIndexList;

    ExactSearchTester gtTester;

    private IEnumerable<int> TestKValues => Enumerable.Range(groundTruthK / 5, 501 / 5).Select(a => a * 5);

    public ReducedDimensionHnswTester(string datasetName, string inputPath, string inputReducedDataFileName, bool debugMode, ExactSearchTester gtTester)
        : base(datasetName, debugMode)
    {
        this.inputPath = inputPath;
        this._inputReducedDataFileName = inputReducedDataFileName;
        this.gtTester = gtTester;
    }

    public void LoadDatabase(int? maxDataSize)
    {
        var reducedDimensionDatasetFileName = Path.Join(inputPath, _inputReducedDataFileName);
        ExactSearchTester.PrepareDataset(true, reducedDimensionDatasetFileName, out _embeddedVectorsListReduced, out _textListReduced, maxDataSize);
    }

    public void Run(int maxDegreeOfParallelism, (int candidateIndex, float Score)[][] groundTruthResults, TimeSpan groundTruthRuntime, int? maxDataSize = null)
    {
        maxDataSize ??= _embeddedVectorsListOriginal.Length;

        LoadDatabase(maxDataSize);

        var dataSizes = Enumerable
            .Range(1, 10)
            .Select(a => a * 10000)
            .ToArray();

        //TestExactScoreAndOrderBySort(maxDegreeOfParallelism, _embeddedVectorsListReduced, groundTruthK, datasetName, seedsIndexList, groundTruthResults, groundTruthRuntime);
        //TestExactScoreAndOrderBySortExtendedK(maxDegreeOfParallelism, _embeddedVectorsListReduced, groundTruthK, datasetName, seedsIndexList, _embeddedVectorsListOriginal, groundTruthResults, groundTruthRuntime, 0.95f);

        //TestPriorityQueueScoreAndSort(maxDegreeOfParallelism, _embeddedVectorsListOriginal, groundTruthK, datasetName, seedsIndexList, groundTruthResults, groundTruthRuntime);

        TestPriorityQueueScoreAndSort(maxDegreeOfParallelism, _embeddedVectorsListReduced, groundTruthK, DatasetName, seedsIndexList, groundTruthResults, groundTruthRuntime);
        TestPriorityQueueScoreAndSortExtendedK(maxDegreeOfParallelism, DatasetName, _embeddedVectorsListReduced, groundTruthK, seedsIndexList, _embeddedVectorsListOriginal, groundTruthResults, groundTruthRuntime, 0.95f);

        foreach (var size in dataSizes)
        {
            if (size > maxDataSize)
            {
                break;
            }

            var partialOriginalDataArray = _embeddedVectorsListOriginal.Take(size).ToArray();
            var partialReducedDataArray = _embeddedVectorsListReduced.Take(size).ToArray();
            BuildGroundTruthResults(maxDegreeOfParallelism, groundTruthK, DatasetName, partialOriginalDataArray, seedsIndexList, out groundTruthResults, out groundTruthRuntime, gtTester.UseHeapSort);

            TestPriorityQueueScoreAndSortExtendedK(maxDegreeOfParallelism, DatasetName, partialReducedDataArray, groundTruthK, seedsIndexList, partialOriginalDataArray, groundTruthResults, groundTruthRuntime, 0.85f);
            TestPriorityQueueScoreAndSortExtendedK(maxDegreeOfParallelism, DatasetName, partialReducedDataArray, groundTruthK, seedsIndexList, partialOriginalDataArray, groundTruthResults, groundTruthRuntime, 0.95f);
            TestPriorityQueueScoreAndSortExtendedK(maxDegreeOfParallelism, DatasetName, partialReducedDataArray, groundTruthK, seedsIndexList, partialOriginalDataArray, groundTruthResults, groundTruthRuntime, 0.99f);
        }
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
        var heap = new ScoreAndSortHeapSort(maxDegreeOfParallelism, k, datasetName, embeddedVectorsList, SIMDCosineSimilarityVectorsScoreForUnits);
        heap.Run(seedsIndexList);
        heap.Evaluate($"Pr. Queue [k={k}]", seedsIndexList, groundTruthResults, groundTruthElapsedTime);
        PrintDataSampleDebug(heap.Results);
    }

    private void TestPriorityQueueScoreAndSortExtendedK(
        int maxDegreeOfParallelism,
        string datasetName,
        float[][] embeddedVectorsListReduced,
        int groundTruthK,
        int[] seedsIndexList,
        float[][] embeddedVectorsListOriginal,
        (int candidateIndex, float Score)[][] groundTruthResults,
        TimeSpan groundTruthElapsedTime,
        float desiredRecall)
    {
        foreach (var extendedK in TestKValues)
        {
            var heap = new ScoreAndSortHeapSort(maxDegreeOfParallelism, extendedK, datasetName, embeddedVectorsListReduced, SIMDCosineSimilarityVectorsScoreForUnits, embeddedVectorsListOriginal);
            heap.Run(seedsIndexList);
            var r0 = heap.EvaluateScoring(groundTruthResults);

            heap.ReScore(seedsIndexList, embeddedVectorsListOriginal, groundTruthK);

            var r = heap.EvaluateScoring(groundTruthResults);
            Console.WriteLine($"TestPriorityQueueScoreAndSortExtendedK: Look for {desiredRecall} recall at k={extendedK}, found {r0} or {r} recall when sorted by original dim");
            if (r > desiredRecall)
            {
                heap.Evaluate($"Pr.Queue [k={groundTruthK}/{extendedK}]", seedsIndexList, groundTruthResults, groundTruthElapsedTime);
                PrintDataSampleDebug(heap.Results);
                break;
            }

            Console.WriteLine($"\t: k={extendedK}, {r} recall");
        }
    }

}
