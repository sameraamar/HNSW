// See https://aka.ms/new-console-template for more information

using HNSW;

internal class CppHnswTester : HnswBaseTester
{
    private readonly int _maxDegreeOfParallelismBuild = Environment.ProcessorCount;

    private float[][] _embeddedVectorsListOriginal => gtTester._embeddedVectorsListOriginal;
    //private List<string> _textListOriginal;
    //private List<string> _textListReduced;      

    private int groundTruthK => gtTester.GroundTruthK;
    //private const int seedsCount = 100;
    private int[] seedsIndexList => gtTester.seedsIndexList;

    ExactSearchTester gtTester;

    public CppHnswTester(ExactSearchTester gtTester, int mParam, int efConstruction, string outputPath)
        : base(gtTester.DatasetName, gtTester.DebugMode, mParam, efConstruction, outputPath)
    {
        this.gtTester = gtTester;
    }


    public void Run(int maxDegreeOfParallelism, (int candidateIndex, float Score)[][] groundTruthResults, TimeSpan groundTruthRuntime, int? maxDataSize = null)
    {
        maxDataSize = _embeddedVectorsListOriginal.Length;

        var dataSizes = Enumerable
            .Range(1, 10)
            .Select(a => a * 10000)
            .ToArray();

        TestHNSWScoreAndSortCpp(maxDegreeOfParallelism, _embeddedVectorsListOriginal, groundTruthK, DatasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);
        TestHNSWScoreAndSortCpp(0.90f, maxDegreeOfParallelism, _embeddedVectorsListOriginal, DatasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);

        //TestExactScoreAndOrderBySort(maxDegreeOfParallelism, _embeddedVectorsListReduced, groundTruthK, datasetName, seedsIndexList, groundTruthResults, groundTruthRuntime);
        //TestExactScoreAndOrderBySortExtendedK(maxDegreeOfParallelism, _embeddedVectorsListReduced, groundTruthK, datasetName, seedsIndexList, _embeddedVectorsListOriginal, groundTruthResults, groundTruthRuntime, 0.95f);

        //TestPriorityQueueScoreAndSort(maxDegreeOfParallelism, _embeddedVectorsListOriginal, groundTruthK, datasetName, seedsIndexList, groundTruthResults, groundTruthRuntime);

        // HNSW C++
        foreach (var size in dataSizes)
        {
            var partialOriginalDataArray = _embeddedVectorsListOriginal.Take(size).ToArray();
            BuildGroundTruthResults(maxDegreeOfParallelism, groundTruthK, DatasetName, partialOriginalDataArray, seedsIndexList, out groundTruthResults, out groundTruthRuntime);
            TestHNSWScoreAndSortCpp(0.85f, maxDegreeOfParallelism, partialOriginalDataArray, DatasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);
            TestHNSWScoreAndSortCpp(0.95f, maxDegreeOfParallelism, partialOriginalDataArray, DatasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);
            TestHNSWScoreAndSortCpp(0.99f, maxDegreeOfParallelism, partialOriginalDataArray, DatasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);
        }
    }

    protected void BuildGroundTruthResults(
        int maxDegreeOfParallelism,
        int groundTruthK,
        string datasetName,
        float[][] embeddedVectorsListOriginal,
        int[] seedsIndexList,
        out (int candidateIndex, float Score)[][] groundTruthResults,
        out TimeSpan groundTruthElapsedTime)
    {
        var ss = new ScoreAndSortHeapSort(maxDegreeOfParallelism, groundTruthK, datasetName, embeddedVectorsListOriginal, SIMDCosineSimilarityVectorsScoreForUnits);
        ss.Run(seedsIndexList);
        PrintDataSampleDebug(ss.Results);

        groundTruthResults = ss.Results;
        groundTruthElapsedTime = ss.ElapsedTime;

        ss.Evaluate($"GT OrderBySort [k={groundTruthK}]", seedsIndexList, groundTruthResults, groundTruthElapsedTime);
    }

    private void TestHNSWScoreAndSortCpp(int maxDegreeOfParallelism,
        IEnumerable<float[]> embeddedVectorsList,
        int k,
        string datasetName,
        string outputPath,
        int mParam,
        int efConstruction,
        int[] seedsIndexList,
        (int candidateIndex, float Score)[][] groundTruthResults,
        TimeSpan groundTruthElapsedTime)
    {
        var inputDataList = embeddedVectorsList.ToArray();
        using (var hnsw = new ScoreAndSortHNSWCpp( _maxDegreeOfParallelismBuild, maxDegreeOfParallelism, k, datasetName, inputDataList, SIMDCosineDistanceVectorsScoreForUnits, DebugMode, true))
        {
            hnsw.Init(outputPath, inputDataList.Length, mParam, efConstruction);
            hnsw.Run(seedsIndexList);
            hnsw.Evaluate($"HNSW-C++ [k={k}]", seedsIndexList, groundTruthResults, groundTruthElapsedTime);
            PrintDataSampleDebug(hnsw.Results);
        }
    }

    private void TestHNSWScoreAndSortCpp(
        float desiredRecall,
        int maxDegreeOfParallelism,
        float[][] inputDataList,
        string datasetName,
        string outputPath,
        int mParam,
        int efConstruction, int[] seedsIndexList,
        (int candidateIndex, float Score)[][] groundTruthResults,
        TimeSpan groundTruthElapsedTime)
    {
        var kValues = Enumerable.Range(0, 40).Select(a => (1 + a * 0.5) * groundTruthResults[0].Length).Select(a => (int)a);
        foreach (var k in kValues)
        {
            using (var hnsw = new ScoreAndSortHNSWCpp(_maxDegreeOfParallelismBuild, maxDegreeOfParallelism, k, datasetName, inputDataList, SIMDCosineDistanceVectorsScoreForUnits, DebugMode, false))
            {
                hnsw.Init(outputPath, inputDataList.Length, mParam, efConstruction);
                hnsw.Run(seedsIndexList);

                var r = 0f;
                if ((r = hnsw.EvaluateScoring(groundTruthResults)) > desiredRecall)
                {
                    hnsw.Evaluate($"HNSW-C++ [k={k}]", seedsIndexList, groundTruthResults, groundTruthElapsedTime);
                    PrintDataSampleDebug(hnsw.Results);
                    break;
                }

                Console.WriteLine($"\tTestHNSWScoreAndSortCpp: Look for {desiredRecall} recall. found {r} at k={k}");
            }
        }
    }

}

internal class CSharpHnswTester : HnswBaseTester
{
    //#if true
    //    private string datasetName = "Random";

    //    private string inputPath = @"C:\temp\hnsw\Random";
    //    private string outputPath = @"C:\temp\hnsw\Random\results";

    //    private string _inputReducedDataFileName = @"1mcosine-128.npz";
    //    private string _inputOriginalDataFileName = @"1mcosine-1024.csv";
    //#else
    //    private string datasetName = "Wines";
    //    private string inputPath = @"C:\Users\saaamar\OneDrive - Microsoft\temp\hnsw-benchmark\datasets\wines";
    //    private string outputPath = @"c:/temp/hnsw/results";

    //    private string _inputReducedDataFileName = @"wines120kcosine-128.npz";
    //    private string _inputOriginalDataFileName = @"wines120kcosine-1024.npz";
    //#endif


    private float[][] _embeddedVectorsListOriginal;
    //private List<string> _textListOriginal;
    //private List<string> _textListReduced;      

    private int groundTruthK;
    //private const int seedsCount = 100;
    private int[] seedsIndexList;

    private readonly ExactSearchTester gtTester;

    private IEnumerable<int> TestKValues => Enumerable.Range(groundTruthK / 5, 501 / 5).Select(a => a * 5);

    public CSharpHnswTester(ExactSearchTester gtTester, int mParam, int efConstruction, string outputPath)
        :base(gtTester.DatasetName, gtTester.DebugMode, mParam, efConstruction, outputPath)
    {
        groundTruthK = gtTester.GroundTruthK;
        _embeddedVectorsListOriginal = gtTester._embeddedVectorsListOriginal;
        seedsIndexList = gtTester.seedsIndexList;

        this.gtTester = gtTester;
    }

    //
    //private (int candidateIndex, float Score)[][] _groundTruthResults;
    //private TimeSpan _groundTruthElapsedTime;

    public void Run(int maxDegreeOfParallelism, (int candidateIndex, float Score)[][] groundTruthResults, TimeSpan groundTruthRuntime, int? maxDataSize = null)
    { 
        maxDataSize = _embeddedVectorsListOriginal.Length;

        var dataSizes = Enumerable
            .Range(1, 10)
            .Select(a => a * 5000)
            .ToArray();

        //TestHNSWScoreAndSortCpp(maxDegreeOfParallelism, _embeddedVectorsListOriginal, groundTruthK, DatasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);
        TestHNSWScoreAndSortCSharp(maxDegreeOfParallelism, _embeddedVectorsListOriginal, groundTruthK, DatasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);

        //TestExactScoreAndOrderBySort(maxDegreeOfParallelism, _embeddedVectorsListReduced, groundTruthK, datasetName, seedsIndexList, groundTruthResults, groundTruthRuntime);
        //TestExactScoreAndOrderBySortExtendedK(maxDegreeOfParallelism, _embeddedVectorsListReduced, groundTruthK, datasetName, seedsIndexList, _embeddedVectorsListOriginal, groundTruthResults, groundTruthRuntime, 0.95f);

        //TestPriorityQueueScoreAndSort(maxDegreeOfParallelism, _embeddedVectorsListOriginal, groundTruthK, datasetName, seedsIndexList, groundTruthResults, groundTruthRuntime);
        
        /*
        TestPriorityQueueScoreAndSort(maxDegreeOfParallelism, _embeddedVectorsListReduced, groundTruthK, DatasetName, seedsIndexList, groundTruthResults, groundTruthRuntime);
        TestPriorityQueueScoreAndSortExtendedK(maxDegreeOfParallelism, DatasetName, _embeddedVectorsListReduced, groundTruthK, seedsIndexList, _embeddedVectorsListOriginal, groundTruthResults, groundTruthRuntime, 0.95f);
        */

        // HNSW C#
        {
            BuildGroundTruthResults(maxDegreeOfParallelism, groundTruthK, DatasetName, _embeddedVectorsListOriginal, seedsIndexList, out groundTruthResults, out groundTruthRuntime, gtTester.UseHeapSort);
            TestHNSWScoreAndSortCSharp(0.95f, maxDegreeOfParallelism, _embeddedVectorsListOriginal, DatasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);
            TestHNSWScoreAndSortCSharp(0.85f, maxDegreeOfParallelism, _embeddedVectorsListOriginal, DatasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);
        }

        //// HNSW C++
        //foreach (var size in dataSizes)
        //{
        //    var partialOriginalDataArray = _embeddedVectorsListOriginal.Take(size).ToArray();
        //    BuildGroundTruthResults(maxDegreeOfParallelism, groundTruthK, DatasetName, partialOriginalDataArray, seedsIndexList, out groundTruthResults, out groundTruthRuntime);
        //    TestHNSWScoreAndSortCpp(0.85f, maxDegreeOfParallelism, partialOriginalDataArray, DatasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);
        //    TestHNSWScoreAndSortCpp(0.95f, maxDegreeOfParallelism, partialOriginalDataArray, DatasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);
        //    TestHNSWScoreAndSortCpp(0.99f, maxDegreeOfParallelism, partialOriginalDataArray, DatasetName, outputPath, mParam, efConstruction, seedsIndexList, groundTruthResults, groundTruthRuntime);
        //}

        /*
        foreach (var size in dataSizes)
        {
            var partialOriginalDataArray = _embeddedVectorsListOriginal.Take(size).ToArray();
            var partialReducedDataArray = _embeddedVectorsListReduced.Take(size).ToArray();
            BuildGroundTruthResults(maxDegreeOfParallelism, groundTruthK, DatasetName, partialOriginalDataArray, seedsIndexList, out groundTruthResults, out groundTruthRuntime);

            TestPriorityQueueScoreAndSortExtendedK(maxDegreeOfParallelism, DatasetName, partialReducedDataArray, groundTruthK, seedsIndexList, partialOriginalDataArray, groundTruthResults, groundTruthRuntime, 0.85f);
            TestPriorityQueueScoreAndSortExtendedK(maxDegreeOfParallelism, DatasetName, partialReducedDataArray, groundTruthK, seedsIndexList, partialOriginalDataArray, groundTruthResults, groundTruthRuntime, 0.95f);
            TestPriorityQueueScoreAndSortExtendedK(maxDegreeOfParallelism, DatasetName, partialReducedDataArray, groundTruthK, seedsIndexList, partialOriginalDataArray, groundTruthResults, groundTruthRuntime, 0.99f);
        }
        */
    }
    /*
    private void BuildGroundTruthResults(
        int maxDegreeOfParallelism,
        int groundTruthK,
        string datasetName,
        float[][] embeddedVectorsListOriginal,
        int[] seedsIndexList,
        out (int candidateIndex, float Score)[][] groundTruthResults,
        out TimeSpan groundTruthElapsedTime)
    {
        var ss = new ScoreAndSortHeapSort(maxDegreeOfParallelism, groundTruthK, datasetName, embeddedVectorsListOriginal, SIMDCosineSimilarityVectorsScoreForUnits);
        ss.Run(seedsIndexList);
        PrintDataSampleDebug(ss.Results);

        groundTruthResults = ss.Results;
        groundTruthElapsedTime = ss.ElapsedTime;

        ss.Evaluate($"GT OrderBySort [k={groundTruthK}]", seedsIndexList, groundTruthResults, groundTruthElapsedTime);
    }
    */
    private void TestHNSWScoreAndSortCpp(int maxDegreeOfParallelism,
        IEnumerable<float[]> embeddedVectorsList, 
        int k, 
        string datasetName, 
        string outputPath, 
        int mParam, 
        int efConstruction, 
        int[] seedsIndexList,
        (int candidateIndex, float Score)[][] groundTruthResults,
        TimeSpan groundTruthElapsedTime)
    {
        var inputDataList = embeddedVectorsList.ToArray();
        using (var hnsw = new ScoreAndSortHNSWCpp( maxDegreeOfParallelism, maxDegreeOfParallelism, k, datasetName, inputDataList, SIMDCosineDistanceVectorsScoreForUnits, DebugMode, true))
        {
            hnsw.Init(outputPath, inputDataList.Length, mParam, efConstruction);
            hnsw.Run(seedsIndexList);
            hnsw.Evaluate($"HNSW-C++ [k={k}]", seedsIndexList, groundTruthResults, groundTruthElapsedTime);
            PrintDataSampleDebug(hnsw.Results);
        }
    }

    private void TestHNSWScoreAndSortCpp(
        float desiredRecall, 
        int maxDegreeOfParallelism, 
        IEnumerable<float[]> embeddedVectorsList, 
        string datasetName, 
        string outputPath, 
        int mParam, 
        int efConstruction, int[] seedsIndexList, 
        (int candidateIndex, float Score)[][] groundTruthResults,
        TimeSpan groundTruthElapsedTime)
    {
        var inputDataList = embeddedVectorsList.ToArray();

        (int candidateIndex, float Score)[][]? results = null;
        string header = "";
        string msg = "";
        foreach (var i in Enumerable.Range(groundTruthResults[0].Length / 5, 100))
        {
            var k = i * 5;
            using (var hnsw = new ScoreAndSortHNSWCpp(maxDegreeOfParallelism, maxDegreeOfParallelism, k, datasetName, inputDataList, SIMDCosineDistanceVectorsScoreForUnits, DebugMode, false))
            {
                hnsw.Init(outputPath, inputDataList.Length, mParam, efConstruction);
                hnsw.Run(seedsIndexList);
                results = hnsw.Results;
                (header, msg) = hnsw.Evaluate($"HNSW-C++ [k={k}]", seedsIndexList, groundTruthResults, groundTruthElapsedTime, false, false);

                var r = 0f;
                if ((r = hnsw.EvaluateScoring(groundTruthResults)) > desiredRecall)
                {
                    break;
                }

                Console.WriteLine($"\tTestHNSWScoreAndSortCpp: Look for {desiredRecall} recall. Found {r} @ k={k}");
            }
        }

        if (results != null)
        {
            Console.WriteLine(header);
            Console.WriteLine(msg);
            PrintDataSampleDebug(results);
        }
    }

    private void TestHNSWScoreAndSortCSharp(int maxDegreeOfParallelism, float[][] embeddedVectorsListOriginal, int k,
        string datasetName, string outputPath, int mParam, int efConstruction, int[] seedsIndexList,
        (int candidateIndex, float Score)[][] groundTruthResults, TimeSpan groundTruthElapsedTime)
    {
        var hnsw = new ScoreAndSortHNSW(maxDegreeOfParallelism, k, datasetName, embeddedVectorsListOriginal, SIMDCosineDistanceVectorsScoreForUnits);
        hnsw.Init(outputPath, mParam, efConstruction);
        hnsw.Run(seedsIndexList);
        hnsw.Evaluate($"HNSW-C# [k={k}]", seedsIndexList, groundTruthResults, groundTruthElapsedTime);
        PrintDataSampleDebug(hnsw.Results);
    }


    private void TestHNSWScoreAndSortCSharp(
        float desiredRecall, 
        int maxDegreeOfParallelism, 
        IEnumerable<float[]> embeddedVectorsList, 
        string datasetName, 
        string outputPath, 
        int mParam, 
        int efConstruction, 
        int[] seedsIndexList,
        (int candidateIndex, float Score)[][] groundTruthResults,
        TimeSpan groundTruthElapsedTime)
    {
        var inputDataList = embeddedVectorsList.ToArray();
        var kValues = Enumerable.Range(1, 10).Select(a => a * groundTruthResults[0].Length);
        (int candidateIndex, float Score)[][]? results = null;
        string header = "";
        string msg = "";

        foreach (var k in kValues)
        {
            var hnsw = new ScoreAndSortHNSW(maxDegreeOfParallelism, k, datasetName, inputDataList, SIMDCosineDistanceVectorsScoreForUnits);
            {
                hnsw.Init(outputPath, mParam, efConstruction);
                hnsw.Run(seedsIndexList);

                results = hnsw.Results;
                (header, msg) = hnsw.Evaluate($"HNSW-C# [k={k}]", seedsIndexList, groundTruthResults, groundTruthElapsedTime, false, false);

                var r = 0f;
                if ((r = hnsw.EvaluateScoring(groundTruthResults)) > desiredRecall)
                {
                    break;
                }

                Console.WriteLine($"\tTestHNSWScoreAndSortC#: Look for {desiredRecall} recall. Found {r} @ k={k}");
            }
        }

        if (results != null)
        {
            Console.WriteLine(header);
            Console.WriteLine(msg);
            PrintDataSampleDebug(results);
        }
    }


    private void TestPriorityQueueScoreAndSort1(
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

    private void TestPriorityQueueScoreAndSortExtendedK1(
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

    private void TestExactScoreAndOrderBySort(int maxDegreeOfParallelism, float[][] embeddedVectorsListReduced,
        int groundTruthK,
        string datasetName, int[] seedsIndexList, (int candidateIndex, float Score)[][] groundTruthResults,
        TimeSpan groundTruthElapsedTime)
    {
        var exactSearchReduced = new ScoreAndSortExact(maxDegreeOfParallelism, groundTruthK, datasetName, embeddedVectorsListReduced, SIMDCosineSimilarityVectorsScoreForUnits);
        exactSearchReduced.Run(seedsIndexList);
        exactSearchReduced.Evaluate($"OrderBySort [k={groundTruthK}]", seedsIndexList, groundTruthResults, groundTruthElapsedTime);
        PrintDataSampleDebug(exactSearchReduced.Results);
    }

    private void TestExactScoreAndOrderBySortExtendedK(
        int maxDegreeOfParallelism,
        float[][] embeddedVectorsListReduced,
        int groundTruthK,
        string datasetName, int[] seedsIndexList,
        float[][] embeddedVectorsListOriginal, 
        (int candidateIndex, float Score)[][] groundTruthResults,
        TimeSpan groundTruthElapsedTime,
        float desiredRecall)
    {
        foreach (var extendedK in TestKValues)
        {
            var exactSearchReduced = new ScoreAndSortExact(maxDegreeOfParallelism, extendedK, datasetName, embeddedVectorsListReduced, SIMDCosineSimilarityVectorsScoreForUnits);
            exactSearchReduced.Run(seedsIndexList);
            exactSearchReduced.ReScore(seedsIndexList, embeddedVectorsListOriginal, groundTruthK);

            var r = 0f;  
            if ((r = exactSearchReduced.EvaluateScoring(groundTruthResults)) > desiredRecall)
            {
                exactSearchReduced.Evaluate($"OrderBySort [k={groundTruthK}/{extendedK}]", seedsIndexList, groundTruthResults, groundTruthElapsedTime);
                PrintDataSampleDebug(exactSearchReduced.Results);
                break;
            }

            Console.WriteLine($"\tTestExactScoreAndOrderBySortExtendedK: Look for {desiredRecall} recall. Found {r} @ k={extendedK}");
        }
    }

}