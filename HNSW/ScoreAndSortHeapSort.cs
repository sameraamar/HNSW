
namespace HNSW
{
    internal class ScoreAndSortHeapSort : ScoreAndSortBase
    {
        public ScoreAndSortHeapSort(int maxDegreeOfParallelism, int maxScoredItems, string datasetName, float[][] embeddedVectorsList, Func<float[], float[], float> distanceFunction)
            : base(maxDegreeOfParallelism, maxScoredItems, datasetName, embeddedVectorsList, distanceFunction)
        {
        }

        public override (int candidateIndex, float Score)[][] Run(int[] seedsIndexList)
        {
            var (elapsedTime, results) = CalculateScoresForSeeds(seedsIndexList);
            Console.WriteLine($"[{this.GetType().Name}] Average per seed: {1f * elapsedTime / seedsIndexList.Length:F2} ms");
            return results;
        }

        //private (long, (int candidateIndex, float Score)[][]) CalculateScoresForSeeds(float[][] seedsVectorList, int[] seedsIndexList, List<float[]> embeddedVectorsList)
        //{
        //    var results = new (int candidateIndex, float Score)[seedsVectorList.Length][];
        //    var sw = Stopwatch.StartNew();
        //    seedsIndexList
        //        .AsParallel()
        //        .WithDegreeOfParallelism(MaxDegreeOfParallelism)
        //        .ForAll(i =>
        //        {
        //            var seed = seedsVectorList[i];
        //            results[i] = CalculateScoresPerSeed(seed, embeddedVectorsList).ToArray();
        //        });

        //    return (sw.ElapsedMilliseconds, results);
        //}

        protected override IEnumerable<(int candidateIndex, float Score)> CalculateScoresPerSeed(int seedIndex)
        {
            var seed = EmbeddedVectorsList[seedIndex];

            var scoresPerSeed = EmbeddedVectorsList
                .Select((candidateVector, candidateIndex) => (candidateIndex, Score: DistanceFunction(seed, candidateVector)));

            return GetTopScoresPriorityQueueSort<int>(scoresPerSeed, MaxScoredItems);
        }

        protected IEnumerable<(int candidateIndex, float Score)> CalculateScoresPerSeed1(int seedIndex, List<float[]> embeddedVectorsList)
        {
            var seed = embeddedVectorsList[seedIndex];

            var pq = new PriorityQueue<int, float>();
            var scoresPerSeed = embeddedVectorsList
                .Select((candidateVector, candidateIndex) => (candidateIndex, Score: DistanceFunction(seed, candidateVector)));
            
            pq.EnqueueRange(scoresPerSeed);

            int k = MaxScoredItems;
            while (k > 0 && pq.TryDequeue(out var element, out var score))
            {
                yield return (element, score);
                k--;
            }
        }

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
