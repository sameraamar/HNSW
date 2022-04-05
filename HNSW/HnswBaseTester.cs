// See https://aka.ms/new-console-template for more information

using HNSW;
using Microsoft.Xbox.Recommendations.Modeling.ETS.Instances.Common;
using System.Runtime.CompilerServices;


internal abstract class BaseTester
{
    public string DatasetName { get; }
    public bool DebugMode { get; }

    protected BaseTester(string datasetName, bool debugMode)
    {
        DatasetName = datasetName;
        DebugMode = debugMode;
    }

    protected void PrintDataSampleDebug((int candidateIndex, float Score)[][] results)
    {
        if (!DebugMode)
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected float SIMDCosineSimilarityVectorsScoreForUnits(float[] l, float[] r)
    {
        return VectorUtilities.SIMDCosineSimilarityVectorsScoreForUnits(l, r);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected float SIMDCosineDistanceVectorsScoreForUnits(float[] l, float[] r)
    {
        return 1F - VectorUtilities.SIMDCosineSimilarityVectorsScoreForUnits(l, r);
    }

    protected void BuildGroundTruthResults(
        int maxDegreeOfParallelism,
        int groundTruthK,
        string datasetName,
        float[][] embeddedVectorsListOriginal,
        int[] seedsIndexList,
        out (int candidateIndex, float Score)[][] groundTruthResults,
        out TimeSpan groundTruthElapsedTime,
        bool usePriorityQueue)
    {
        ScoreAndSortBase ss;
        if (usePriorityQueue)
        {
            ss = new ScoreAndSortHeapSort(maxDegreeOfParallelism, groundTruthK, datasetName, embeddedVectorsListOriginal, SIMDCosineSimilarityVectorsScoreForUnits);
        }
        else
        {
            ss = new ScoreAndSortExact(maxDegreeOfParallelism, groundTruthK, datasetName, embeddedVectorsListOriginal, SIMDCosineSimilarityVectorsScoreForUnits);
        }
            
        ss.Run(seedsIndexList);
        
        groundTruthResults = ss.Results;
        groundTruthElapsedTime = ss.ElapsedTime;

        PrintDataSampleDebug(groundTruthResults);
        
        ss.Evaluate($"GT OrderBySort [k={groundTruthK}]", seedsIndexList, groundTruthResults, groundTruthElapsedTime);
    }
}

internal abstract class HnswBaseTester : BaseTester
{
    protected string outputPath { get; }
    protected int mParam { get; }
    protected int efConstruction { get; }

    protected HnswBaseTester(string datasetName, bool debugMode, int mParam, int efConstruction, string outputPath)
       : base(datasetName, debugMode)
    {
        this.outputPath = outputPath;
        this.mParam = mParam;
        this.efConstruction = efConstruction;
    }
}
