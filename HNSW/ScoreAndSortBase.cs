
using System.Diagnostics;

namespace HNSW
{
    internal abstract class ScoreAndSortBase
    {
        protected int MaxDegreeOfParallelism { get; }
        protected int MaxScoredItems { get; }

        protected Func<float[], float[], float> DistanceFunction { get; }

        protected List<float[]> EmbeddedVectorsList { get; set; }

        protected string DatasetName { get; set; }

        protected float DistanceFunctionByIndex(int u, int v)
        {
            return DistanceFunction(EmbeddedVectorsList[u], EmbeddedVectorsList[v]);
        }

        protected ScoreAndSortBase(int maxDegreeOfParallelism, int maxScoredItems, string datasetName, List<float[]> embeddedVectorsList, Func<float[], float[], float> distanceFunction)
        {
            MaxDegreeOfParallelism = maxDegreeOfParallelism;
            MaxScoredItems = maxScoredItems;
            DistanceFunction = distanceFunction;
            EmbeddedVectorsList = embeddedVectorsList;
            DatasetName = datasetName;
        }


        protected (long, (int candidateIndex, float Score)[][]) CalculateScoresForSeeds(int[] seedsIndexList)
        {
            var results = new (int candidateIndex, float Score)[seedsIndexList.Length][];
            var sw = Stopwatch.StartNew();
            seedsIndexList
                .AsParallel()
                .WithDegreeOfParallelism(MaxDegreeOfParallelism)
                .ForAll(i =>
                {
                    results[i] = CalculateScoresPerSeed(seedsIndexList[i]).ToArray();
                });

            return (sw.ElapsedMilliseconds, results);
        }

        protected abstract IEnumerable<(int candidateIndex, float Score)> CalculateScoresPerSeed(int seedIndex);

        public abstract (int candidateIndex, float Score)[][] Run(int[] seedsIndexList);
    }
}
