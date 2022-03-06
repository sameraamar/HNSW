
using System.Collections;
using System.Runtime.Serialization.Formatters.Binary;

namespace HNSW
{
    using System.Diagnostics;

    internal class ScoreAndSortHNSW : ScoreAndSortBase
    {
        private SmallWorld<int, float> World { get; set; }

        public ScoreAndSortHNSW(int maxDegreeOfParallelism, int maxScoredItems, string datasetName, float[][] embeddedVectorsList, Func<float[], float[], float> distanceFunction)
            : base(maxDegreeOfParallelism, maxScoredItems, datasetName, embeddedVectorsList, distanceFunction)
        {
        }

        public void Init(string modelPath, int mParam, int efConstruction)
        {
            BuildFileNames(modelPath, $"{DatasetName}-dim{Dimensionality}-m{mParam}-ef{efConstruction}", (EmbeddedVectorsList.Length, EmbeddedVectorsList[0].Length), out var graphFilename, out var vectorsFileName);

            if (File.Exists(graphFilename))
            {
                LoadWorld(graphFilename, vectorsFileName);
            }
            else
            {
                CreateSmallWorld(graphFilename, vectorsFileName, mParam, efConstruction);
            }
        }

        public override (int candidateIndex, float Score)[][] Run(int[] seedsIndexList)
        {
            var (elapsedTime, results) = CalculateScoresForSeeds(seedsIndexList);
            Console.WriteLine($"[{this.GetType().Name}] Average per seed: {1f * elapsedTime / seedsIndexList.Length:F2} ms");
            return results;
        }

        private void CreateSmallWorld(string graphFilename, string vectorsFileName, int mParam, int efConstruction)
        {
            var dataSize = EmbeddedVectorsList.Length;
            var embeddedIndexList = Enumerable.Range(0, dataSize).ToArray();

            var p = new SmallWorld<int, float>.Parameters()
            {
                EnableDistanceCacheForConstruction = true,
                InitialDistanceCacheSize = dataSize,
                NeighbourHeuristic = NeighbourSelectionHeuristic.SelectHeuristic,
                KeepPrunedConnections = true,
                ExpandBestSelection = true,
                M = mParam,
                ConstructionPruning = efConstruction,
            };

            World = new SmallWorld<int, float>(
                DistanceFunctionByIndex,
                DefaultRandomGenerator.Instance,
                p);
            var clock = Stopwatch.StartNew();

            var handled = 0;
            var i = 0;

            var SampleIncrSize = 5_000;
            IProgressReporter progressBar = new ProgressReporter(clock);
            var numberOfIterations = Math.Ceiling(1.0 * (dataSize - handled) / SampleIncrSize);

            while (handled < dataSize)
            {
                var incrSize = dataSize - handled;
                if (incrSize > SampleIncrSize)
                {
                    incrSize = SampleIncrSize;
                }

                Console.WriteLine(
                    $"Iteration ({i + 1} / {numberOfIterations}): Adding {incrSize} items. New world size will be {handled + incrSize}.");

                World.AddItems(embeddedIndexList.Skip(handled).Take(incrSize).ToArray(), progressBar);
                handled += incrSize;
                i++;
            }

            var dim = EmbeddedVectorsList[0].Length;
            SaveWorld(graphFilename, vectorsFileName, World, EmbeddedVectorsList, (dataSize, dim));
        }

        private void LoadWorld(string graphFilename, string vectorsFileName)
        {
            Stopwatch clock;

            Console.Write("Loading HNSW graph... ");
            clock = Stopwatch.StartNew();
            BinaryFormatter formatter = new BinaryFormatter();
            
            var data =
                (ArrayList)formatter.Deserialize(
                    new MemoryStream(File.ReadAllBytes(vectorsFileName)));

            if (data == null)
            {
                throw new Exception("data found is null or empty!");
            }

            float[][]? temp = data[2] as float[][];
            if (temp == null)
            {
                throw new Exception("data contains null array!");
            }

            EmbeddedVectorsList = temp;

            var embeddedIndexList = Enumerable.Range(0, EmbeddedVectorsList.Length).ToArray();

            using (var f = File.OpenRead(graphFilename))
            {
                World = SmallWorld<int, float>.DeserializeGraph(embeddedIndexList, DistanceFunctionByIndex, DefaultRandomGenerator.Instance, f);
            }

            Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");
        }

        private static void SaveWorld(string graphFilename, string vectorsFileName, SmallWorld<int, float> world, IList<float[]> catalogItems, (int vcount, int vsize) shape)
        {
            var clock = Stopwatch.StartNew();

            Console.WriteLine($"Saving HNSW graph to '${graphFilename}'...  - elapsed {clock.ElapsedMilliseconds / 1000} sec");

            using (StreamWriter sw = new StreamWriter($"{graphFilename}.debug.txt"))
            {
                sw.WriteLine($"{world.Print()}");
                sw.Close();
            }

            //byte[] buffer = ...;

            //short[] samples = new short[buffer.Length];

            //var bytes = new byte[testCatalog.Count * 4];

            //Buffer.BlockCopy(testCatalog.ToArray(), 0, bytes, 0, testCatalog.Count);
            ////            File.WriteAllBytes(vectorsFileName, bytes);

            //using (FileStream fs = File.Create(vectorsFileName))
            //{
            //    //using (ZipArchive zipArchive = new ZipArchive(fs, ZipArchiveMode.Create))
            //    {
            //        //using (Stream stream1 = zipArchive.CreateEntry("arr_0").Open())
            //        {
            //            fs.Write(bytes);
            //        }
            //    }
            //}

            BinaryFormatter formatter = new BinaryFormatter();
            var listToBeSerialized = new ArrayList()
                {shape.vcount, shape.vsize, catalogItems};
            using (FileStream fs = File.Create(vectorsFileName))
            {
                formatter.Serialize(fs, listToBeSerialized);
            }

            //BinaryFormatter formatter = new BinaryFormatter();
            //using (var fileStream = File.Create(vectorsFileName))
            //{
            //    //using (var streamWriter = new BinaryWriter(fileStream))
            //    {
            //        using (var gStreamWrite = new GZipStream(fileStream, CompressionLevel.Optimal))
            //        {
            //            formatter.(gStreamWrite, catalogItems);
            //        }
            //    }
            //}


            //BinaryFormatter formatter2 = new BinaryFormatter();
            //MemoryStream sampleVectorsStream = new MemoryStream();
            //formatter2.Serialize(sampleVectorsStream, testCatalog);
            //File.WriteAllBytes(vectorsFileName, sampleVectorsStream.ToArray());

            //using (var f = File.Open($"{modelPath}.{GraphPathSuffix}", FileMode.Create))
            //{
            //    world.SerializeGraph(f);
            //}

            using (var f = File.Open(graphFilename, FileMode.Create))
            {
                world.SerializeGraph(f);
            }


            Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");

        }

        private static void BuildFileNames(string modelPath, string prefixName, (int vcount, int vsize) shape, out string graphFilename, out string vectorsFileName)
        {
            string VectorsPathSuffix = "vec";
            string GraphPathSuffix = "gf";

            var outputFileNamePrefix = Path.Join(modelPath, $@"{prefixName}-{shape.vcount:D}-{shape.vsize:D}");
            
            vectorsFileName = $"{outputFileNamePrefix}.{VectorsPathSuffix}";
            graphFilename = $"{outputFileNamePrefix}.{GraphPathSuffix}";
        }

        //private (long, (int candidateIndex, float Score)[][]) CalculateScoresForSeeds(int[] seedsIndexList, List<float[]> embeddedVectorsList)
        //{
        //    var results = new (int candidateIndex, float Score)[seedsIndexList.Length][];
        //    var sw = Stopwatch.StartNew();
        //    for (var i = 0; i < seedsIndexList.Length; i++)
        //    {
        //        results[i] = CalculateScoresPerSeed(seedsIndexList[i]).ToArray();
        //    }

        //    return (sw.ElapsedMilliseconds, results);
        //}

        protected override IEnumerable<(int candidateIndex, float Score)> CalculateScoresPerSeed(int seedIndex)
        {
            var res = World.KNNSearch(seedIndex, MaxScoredItems);
            return res.OrderBy(a => a.Distance).Select(r => (r.Id, 1f - r.Distance));
        }
    }
}
