
namespace HNSW
{
    using Microsoft.Xbox.Recommendations.Modeling.ETS.Instances.Common;
    using System.Diagnostics;

    internal abstract class ScoreAndSortBase
    {
        public int MaxDegreeOfParallelism { get; set; }
        protected int MaxScoredItems { get; private set; }

        protected Func<float[], float[], float> DistanceFunction { get; }

        protected float[][] EmbeddedVectorsList { get; set; }

        protected long DataSize => EmbeddedVectorsList.Length;

        protected int Dimensionality => EmbeddedVectorsList[0].Length;

        protected string DatasetName { get; set; }

        protected ScoreAndSortBase(int maxDegreeOfParallelism, int maxScoredItems, string datasetName, float[][] embeddedVectorsList, Func<float[], float[], float> distanceFunction)
        {
            MaxDegreeOfParallelism = maxDegreeOfParallelism;
            MaxScoredItems = maxScoredItems;
            DistanceFunction = distanceFunction;
            EmbeddedVectorsList = embeddedVectorsList;
            DatasetName = datasetName;
        }

        public void Run(int[] seedsIndexList)
        {
            (ElapsedTime, Results) = CalculateScoresForSeeds(seedsIndexList);
        }

        public void ReScore(int[] seedsIndexList, float[][]? UseMeForFinalOrderBy, int newMaxScoredItemsPerSeed)
        {
            if (UseMeForFinalOrderBy != null)
            {
                var reScoreSW = Stopwatch.StartNew();
                
                var results = seedsIndexList
                    .Select((seed, i) =>
                    {
                        var resultsWithNewScores =
                            Results[i]
                                .Select(a => (a.candidateIndex, DistanceFunction(UseMeForFinalOrderBy[seedsIndexList[i]], UseMeForFinalOrderBy[a.candidateIndex])));

                        return GetTopScores(resultsWithNewScores, newMaxScoredItemsPerSeed).ToArray();
                    })
                    .ToArray();

                MaxScoredItems = newMaxScoredItemsPerSeed;
                ElapsedTime += reScoreSW.Elapsed;
                Results = results;
            }
        }

        public (string header, string msg) Evaluate(string label, int[] seedsIndexList, (int candidateIndex, float Score)[][]? groundTruthResults, TimeSpan groundTruthTimeSpan, bool printHeader = false, bool print = true)
        {
            var length = seedsIndexList.Length;
            var runtime = 1f * ElapsedTime.TotalMilliseconds / length;
            var gtRuntime = 1f * groundTruthTimeSpan.TotalMilliseconds / length;
            double speedUp = gtRuntime / runtime;

            var header = $"TypeName\tLabel\tDim\tDataSize\tSeeds Size\tAverage per seed\tAverage GT per seed\tSpeed up";
            var msg = $"[{this.GetType().Name}]\t[{label}]\t{Dimensionality}\t{DataSize}\t{length}\t{runtime:F2}\t{gtRuntime:F2}\t{speedUp:F}";
            if (groundTruthResults != null)
            {
                header += "\tRecall\tWrong Scores";
                var r = EvaluateScoring(groundTruthResults, Results);
                msg += $"\t{r.Recall}\t{r.NotMatchingScores}";
            }

            if (printHeader)
            {
                Console.WriteLine(header);
            }

            if (print)
            {
                Console.WriteLine(msg);
            }

            return (header, msg);
        }

        public (int candidateIndex, float Score)[][] Results { get; private set; }

        public TimeSpan ElapsedTime { get; private set; }

        public (float Recall, int NotMatchingScores) EvaluateScoring((int candidateIndex, float Score)[][] groundTruthResults)
        {
            return EvaluateScoring(groundTruthResults, Results);
        }

        protected float DistanceFunctionByIndex(int u, int v)
        {
            return DistanceFunction(EmbeddedVectorsList[u], EmbeddedVectorsList[v]);
        }

        protected virtual (TimeSpan, (int candidateIndex, float Score)[][]) CalculateScoresForSeeds(int[] seedsIndexList)
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

            return (sw.Elapsed, results);
        }

        protected abstract IEnumerable<(int candidateIndex, float Score)> CalculateScoresPerSeed(int seedIndex);

        #region private

        private static (float Recall, int NotMatchingScores) EvaluateScoring((int candidateIndex, float Score)[][] groundTruthResults, (int candidateIndex, float Score)[][] results)
        {
            var groundTruthK = groundTruthResults[0].Length;

            var hits = CountHitsDictionary(results, groundTruthResults, groundTruthK);
            var recall = 1.0f * hits.Select(c => c.hits).Sum() / groundTruthK / hits.Count;
            var notMatchingScores = hits.Select(c => c.countWrongScores).Sum();

            return (recall, notMatchingScores);
        }


        private static List<(int hits, int countWrongScores)> CountHitsDictionary((int candidateIndex, float Score)[][] recoItemsTnsw, (int candidateIndex, float Score)[][] recoItemsGroundTruth, int? groundTruthK = null)
        {
            var recall = Enumerable.Repeat((0, 0), recoItemsGroundTruth.Length).ToList();

            if (recoItemsTnsw.Length == 0)
            {
                return recall;
            }

            for (int i = 0; i < recoItemsGroundTruth.Length; i++)
            {
                var recommendedItemsPairwise = recoItemsGroundTruth[i];
                var recommendedItemsTnsw = recoItemsTnsw[i];
                var recommendedItemsTnswTopK = groundTruthK.HasValue ? recommendedItemsTnsw.Take(groundTruthK.Value) : recommendedItemsTnsw;

                IEnumerable<(int itemIndex, float itemScore, int gtIndex, float gtScore)> mapIndices =
                    recommendedItemsPairwise
                        .Select(recommendedItem =>
                        {
                            var gt = recommendedItemsTnswTopK
                                .Where(s => s.candidateIndex == recommendedItem.candidateIndex)
                                .FirstOrDefault((-1, 0.0f));

                            return (recommendedItem.candidateIndex, recommendedItem.Score, gt.Item1, gt.Item2);
                        })
                        .ToArray();

                var count = mapIndices.Count(a => a.gtIndex > -1 && a.gtIndex == a.itemIndex);

                var count1 = recommendedItemsPairwise.Count(recommendedItem =>
                {
                    var temp = groundTruthK.HasValue ? recommendedItemsTnsw.Take(groundTruthK.Value) : recommendedItemsTnsw;
                    return temp.Any(s => s.candidateIndex == recommendedItem.candidateIndex);
                });

                Debug.Assert(count == count1);

                var countWrongScores  = mapIndices.Count(a => a.gtIndex != -1 && a.gtIndex == a.itemIndex && !VectorUtilities.CompareAlmostEqual(a.gtScore, a.itemScore));
                var countNotFoundItems  = mapIndices.Count(a => a.gtIndex == -1);
                var countFoundItems = mapIndices.Count(a => a.gtIndex != -1);

                //Debug.Assert(countWrongScores == 0);
                Debug.Assert(countFoundItems + countNotFoundItems == groundTruthK);

                recall[i] = (count, countWrongScores);
            }

            return recall;
        }

        #endregion

        public static IEnumerable<(TId id, float score)> GetTopScores<TId>(IEnumerable<(TId id, float score)> results, int resultsCount)
        {
            return results.OrderByDescending(r => r.score).Take(resultsCount);
        }
    }
}
