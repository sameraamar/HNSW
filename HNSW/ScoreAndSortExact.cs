
namespace HNSW
{
    internal class ScoreAndSortExact : ScoreAndSortBase
    {
        public ScoreAndSortExact(int maxDegreeOfParallelism, int maxScoredItems, string datasetName, float[][] embeddedVectorsList, Func<float[], float[], float> distanceFunction)
        :base(maxDegreeOfParallelism, maxScoredItems, datasetName, embeddedVectorsList, distanceFunction)
        {
        }

        protected override IEnumerable<(int candidateIndex, float Score)> CalculateScoresPerSeed(int seedIndex)
        {
            var seed = EmbeddedVectorsList[seedIndex];

            var scoresPerSeed = EmbeddedVectorsList
                .Select((candidateVector, candidateIndex) => (candidateIndex, Score: DistanceFunction(seed, candidateVector)));

            return GetTopScores(scoresPerSeed, MaxScoredItems);
        }
    }
}
