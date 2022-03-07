
using Microsoft.Xbox.Recommendations.Modeling.ETS.Instances.Common;

namespace HNSW
{
    using System.Diagnostics;

    internal class ScoreAndSortExact : ScoreAndSortBase
    {
        public ScoreAndSortExact(string label, int maxDegreeOfParallelism, int maxScoredItems, string datasetName, float[][] embeddedVectorsList, Func<float[], float[], float> distanceFunction, float[][]? useMeForFinalOrderBy = null, int? evaluationK = null)
        :base(label, maxDegreeOfParallelism, maxScoredItems, datasetName, embeddedVectorsList, distanceFunction)
        {
            UseMeForFinalOrderBy = useMeForFinalOrderBy;
            EvaluationK = evaluationK;

            if (UseMeForFinalOrderBy != null && !EvaluationK.HasValue ||
                UseMeForFinalOrderBy == null && EvaluationK.HasValue)
            {
                throw new Exception("Parameters must be both populated.");
            }
        }

        public int? EvaluationK { get; set; }

        public float[][]? UseMeForFinalOrderBy { get; set; }

        protected override IEnumerable<(int candidateIndex, float Score)> CalculateScoresPerSeed(int seedIndex)
        {
            var seed = EmbeddedVectorsList[seedIndex];

            var scoresPerSeed = EmbeddedVectorsList
                .Select((candidateVector, candidateIndex) => (candidateIndex, Score: DistanceFunction(seed, candidateVector)));

            var results = GetTopScores(scoresPerSeed, MaxScoredItems);

            if (UseMeForFinalOrderBy != null && EvaluationK.HasValue)
            {
                results = GetTopScores(results.Select(a => (a.id, DistanceFunction(UseMeForFinalOrderBy[seedIndex], UseMeForFinalOrderBy[a.id]))), MaxScoredItems);
                results = results.Take(EvaluationK.Value);
            }

            return results;
        }

        public static IEnumerable<(TId id, float score)> GetTopScores<TId>(IEnumerable<(TId id, float score)> results, int resultsCount)
        {
            return results.OrderByDescending(r => r.score).Take(resultsCount);
        }

    }
}
