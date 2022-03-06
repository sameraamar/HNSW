
using Microsoft.Xbox.Recommendations.Modeling.ETS.Instances.Common;

namespace HNSW
{
    using System.Diagnostics;

    internal class ScoreAndSortExact : ScoreAndSortBase
    {
        public ScoreAndSortExact(int maxDegreeOfParallelism, int maxScoredItems, string datasetName, float[][] embeddedVectorsList, Func<float[], float[], float> distanceFunction)
        :base(maxDegreeOfParallelism, maxScoredItems, datasetName, embeddedVectorsList, distanceFunction)
        {
        }

        public override (int candidateIndex, float Score)[][] Run(int[] seedsIndexList)
        {
            var (elapsedTime, results) = CalculateScoresForSeeds(seedsIndexList);
            Console.WriteLine($"[{this.GetType().Name}] Average per seed: {1f * elapsedTime / seedsIndexList.Length:F2} ms");

            return results;
        }

        protected override IEnumerable<(int candidateIndex, float Score)> CalculateScoresPerSeed(int seedIndex)
        {
            var seed = EmbeddedVectorsList[seedIndex];

            var scoresPerSeed = EmbeddedVectorsList
                .Select((candidateVector, candidateIndex) => (candidateIndex, Score: DistanceFunction(seed, candidateVector)));

            return GetTopScores(scoresPerSeed, MaxScoredItems);

            //var seed = embeddedVectorsList[seedIndex];
            //var scoresPerSeed = embeddedVectorsList
            //    .Select((candidateVector, candidateIndex) => (candidateIndex, Score: DistanceFunction(seed, candidateVector)))
            //    .OrderByDescending(e => e.Score)
            //    .Take(MaxScoredItems);

            //return scoresPerSeed;
        }


        public static IEnumerable<(TId id, float score)> GetTopScores<TId>(IEnumerable<(TId id, float score)> results, int resultsCount)
        {
            return results.OrderByDescending(r => r.score).Take(resultsCount);
        }

    }
}
