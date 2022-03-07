
using Microsoft.Xbox.Recommendations.Modeling.ETS.Instances.Common;

namespace HNSW
{
    using System.Diagnostics;

    internal class ScoreAndSortExact : ScoreAndSortBase
    {
        public ScoreAndSortExact(int maxDegreeOfParallelism, int maxScoredItems, string datasetName, float[][] embeddedVectorsList, Func<float[], float[], float> distanceFunction, float[][]? useMeForFinalOrderBy = null)
        :base(maxDegreeOfParallelism, maxScoredItems, datasetName, embeddedVectorsList, distanceFunction)
        {
            UseMeForFinalOrderBy = useMeForFinalOrderBy;
        }

        public float[][]? UseMeForFinalOrderBy { get; set; }

        protected override IEnumerable<(int candidateIndex, float Score)> CalculateScoresPerSeed(int seedIndex)
        {
            var seed = EmbeddedVectorsList[seedIndex];

            var scoresPerSeed = EmbeddedVectorsList
                .Select((candidateVector, candidateIndex) => (candidateIndex, Score: DistanceFunction(seed, candidateVector)));

            var results = GetTopScores(scoresPerSeed, MaxScoredItems);

            if (UseMeForFinalOrderBy != null)
            {
                results = GetTopScores(results.Select(a => (a.id, DistanceFunction(UseMeForFinalOrderBy[seedIndex], UseMeForFinalOrderBy[a.id]))), MaxScoredItems);
            }

            return results;
        }


        public static IEnumerable<(TId id, float score)> GetTopScores<TId>(IEnumerable<(TId id, float score)> results, int resultsCount)
        {
            return results.OrderByDescending(r => r.score).Take(resultsCount);
        }

    }
}
