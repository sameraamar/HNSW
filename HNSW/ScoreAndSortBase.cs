
using System.Diagnostics;
using System.Reflection.Emit;

namespace HNSW
{
    internal abstract class ScoreAndSortBase
    {
        public string Label { get; private set; }
        protected int MaxDegreeOfParallelism { get; }
        protected int MaxScoredItems { get; }

        protected Func<float[], float[], float> DistanceFunction { get; }

        protected float[][] EmbeddedVectorsList { get; set; }

        protected long DataSize => EmbeddedVectorsList.Length;

        protected int Dimensionality => EmbeddedVectorsList[0].Length;

        protected string DatasetName { get; set; }

        protected ScoreAndSortBase(string label, int maxDegreeOfParallelism, int maxScoredItems, string datasetName, float[][] embeddedVectorsList, Func<float[], float[], float> distanceFunction)
        {
            Label = label;
            MaxDegreeOfParallelism = maxDegreeOfParallelism;
            MaxScoredItems = maxScoredItems;
            DistanceFunction = distanceFunction;
            EmbeddedVectorsList = embeddedVectorsList;
            DatasetName = datasetName;
        }

        public (int candidateIndex, float Score)[][] Run(int[] seedsIndexList, (int candidateIndex, float Score)[][]? groundTruthResults = null)
        {
            var (elapsedTime, results) = CalculateScoresForSeeds(seedsIndexList);

            var length = seedsIndexList.Length;
            var msg = $"[{this.GetType().Name}] - [{Label}] Dim = {Dimensionality}, DataSize = {DataSize}, Seeds Size = {length}. Average per seed: {1f * elapsedTime / length:F2} ms";
            if (groundTruthResults != null)
            {
                msg += $". Recall = {EvaluateScoring(groundTruthResults, results)}";
            }
            Console.WriteLine(msg);

            return results;
        }
        
        protected float DistanceFunctionByIndex(int u, int v)
        {
            return DistanceFunction(EmbeddedVectorsList[u], EmbeddedVectorsList[v]);
        }

        protected virtual (long, (int candidateIndex, float Score)[][]) CalculateScoresForSeeds(int[] seedsIndexList)
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

        #region private

        private static float EvaluateScoring((int candidateIndex, float Score)[][] groundTruthResults, (int candidateIndex, float Score)[][] results)
        {
            var groundTruthK = groundTruthResults[0].Length;

            var hits = CountHitsDictionary(results, groundTruthResults, groundTruthK);
            var recall = (float)(1.0 * hits.Sum() / groundTruthK / hits.Count);

            return recall;
        }


        private static List<int> CountHitsDictionary((int candidateIndex, float Score)[][] recoItemsTnsw, (int candidateIndex, float Score)[][] recoItemsGroundTruth, int? groundTruthK = null)
        {
            var recall = Enumerable.Repeat(0, recoItemsGroundTruth.Length).ToList();

            if (recoItemsTnsw.Length == 0)
            {
                return recall;
            }

            for (int i = 0; i < recoItemsGroundTruth.Length; i++)
            {
                var recommendedItemsPairwise = recoItemsGroundTruth[i];
                var recommendedItemsTnsw = recoItemsTnsw[i];

                var count = recommendedItemsPairwise.Count(recommendedItem =>
                {
                    var temp = groundTruthK.HasValue ? recommendedItemsTnsw.Take(groundTruthK.Value) : recommendedItemsTnsw;
                    return temp.Any(s => s.candidateIndex == recommendedItem.candidateIndex);
                });
                recall[i] = count;
            }

            return recall;
        }

        #endregion
    }
}
