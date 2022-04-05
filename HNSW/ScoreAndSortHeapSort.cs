
namespace HNSW
{
    internal class ScoreAndSortHeapSort : ScoreAndSortBase
    {
        public ScoreAndSortHeapSort(int maxDegreeOfParallelism, int maxScoredItems, string datasetName, float[][] embeddedVectorsList, Func<float[], float[], float> distanceFunction, float[][]? useMeForFinalOrderBy = null, int? evaluationK = null)
            : base(maxDegreeOfParallelism, maxScoredItems, datasetName, embeddedVectorsList, distanceFunction)
        {
            UseMeForFinalOrderBy = useMeForFinalOrderBy;
            EvaluationK = evaluationK;
        }

        public int? EvaluationK { get; set; }

        public float[][]? UseMeForFinalOrderBy { get; set; }

        protected override IEnumerable<(int candidateIndex, float Score)> CalculateScoresPerSeed(int seedIndex)
        {
            var seed = EmbeddedVectorsList[seedIndex];

            var scoresPerSeed = EmbeddedVectorsList
                .Select((candidateVector, candidateIndex) => (candidateIndex, Score: DistanceFunction(seed, candidateVector)));

            var results = GetTopScoresPriorityQueueSort<int>(scoresPerSeed, MaxScoredItems);

            if (UseMeForFinalOrderBy != null && EvaluationK.HasValue)
            {
                results = results.Select(a => (a.id, DistanceFunction(UseMeForFinalOrderBy[seedIndex], UseMeForFinalOrderBy[a.id])));
                results = GetTopScoresPriorityQueueSort(results, MaxScoredItems);
                results = results.Take(EvaluationK.Value);
            }

            return results;
        }

        //protected IEnumerable<(int candidateIndex, float Score)> CalculateScoresPerSeed1(int seedIndex, List<float[]> embeddedVectorsList)
        //{
        //    var seed = embeddedVectorsList[seedIndex];

        //    var pq = new PriorityQueue<int, float>();
        //    var scoresPerSeed = embeddedVectorsList
        //        .Select((candidateVector, candidateIndex) => (candidateIndex, Score: DistanceFunction(seed, candidateVector)));
            
        //    pq.EnqueueRange(scoresPerSeed);

        //    int k = MaxScoredItems;
        //    while (k > 0 && pq.TryDequeue(out var element, out var score))
        //    {
        //        yield return (element, score);
        //        k--;
        //    }
        //}

        public static IEnumerable<(TId id, float score)> GetTopScoresPriorityQueueSort<TId>(IEnumerable<(TId id, float score)> results, int resultsCount)
        {
            var pQueue = new PriorityQueue<TId, float>(resultsCount);
            foreach (var stringPlusScore in results)
            {
                if (pQueue.Count == resultsCount)
                {
                    pQueue.EnqueueDequeue(stringPlusScore.id, stringPlusScore.score);
                }
                else
                {
                    pQueue.Enqueue(stringPlusScore.id, stringPlusScore.score);
                }
            }

            return pQueue.UnorderedItems.OrderByDescending(i => i.Priority).Select(a => (a.Element, a.Priority));
        }
    }
}
