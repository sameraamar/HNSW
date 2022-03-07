
using System.Collections;
using System.Runtime.InteropServices;
using System.Runtime.Serialization.Formatters.Binary;
using NumpyIO;


namespace HNSW
{
    using System.Diagnostics;

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct StringInfoA
    {
        [MarshalAs(UnmanagedType.LPStr)] public string f1;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)] public string f2;
    }

    [StructLayout(LayoutKind.Sequential), Serializable]
    public struct ItemAndScore
    {
        [MarshalAs(UnmanagedType.I8)] public long Item;
        [MarshalAs(UnmanagedType.R4)] public float Score;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct SearchResult
    {
        [MarshalAs(UnmanagedType.I8)] public long Id;
        [MarshalAs(UnmanagedType.I8)] public long Size;
    }

    public class Index : IDisposable
    {
#if DEBUG
        internal const string path = $@"..\..\..\..\x64\Debug";
#else
        internal const string path = $@"..\..\..\..\x64\Release";
#endif

        private IntPtr? index;


        [DllImport($@"{path}\HNSWDll.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        public static extern IntPtr Index_Create([MarshalAs(UnmanagedType.LPStr)] string space_name, int dim);
        
        [DllImport($@"{path}\HNSWDll.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Index_Delete(IntPtr index);

        [DllImport($@"{path}\HNSWDll.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void Index_Init(IntPtr index, long maxElements, long M, long efConstruction, long random_seed);

        [DllImport($@"{path}\HNSWDll.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void Index_Save(IntPtr index, [MarshalAs(UnmanagedType.LPStr)] string pathToIndex);

        [DllImport($@"{path}\HNSWDll.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void Index_Load(IntPtr index, [MarshalAs(UnmanagedType.LPStr)] string pathToIndex, long maxElements);

        [DllImport($@"{path}\HNSWDll.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void Index_AddItems(IntPtr index, [MarshalAs(UnmanagedType.LPArray)] float[] input, [MarshalAs(UnmanagedType.LPArray)] long[] ids, int size, int threads = 1);

        [DllImport($@"{path}\HNSWDll.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void Index_Search(IntPtr index, [In, MarshalAs(UnmanagedType.LPArray)] float[,] input, int qsize, int k, [In, Out, MarshalAs(UnmanagedType.LPArray)] SearchResult[] results, int threads = 1);

        [DllImport($@"{path}\HNSWDll.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void Index_Search1(IntPtr index, [In, MarshalAs(UnmanagedType.LPArray)] float[] input, int qsize, int k, [In, Out, MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStruct)] ItemAndScore[] results, [In, Out, MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I8)] long[] rSize, int threads = 1);

        public Index(string space_name, int dim)
        {
            try
            {
                index = Index_Create(space_name, dim);
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                throw;
            }
        }
        
        public void Init(long maxElements, long M, long efConstruction, long random_seed)
        {
            Index_Init(index.Value, maxElements, M, efConstruction, random_seed);
        }

        public void Load(string pathToIndex, long maxElements)
        {
            Index_Load(index.Value, pathToIndex, maxElements);
            //TODO: add verification that dimension is the same
        }

        public void Save(string pathToIndex)
        {
            Index_Save(index.Value, pathToIndex);
        }

        public void AddItems(IEnumerable<float[]> input, IEnumerable<long> ids, int length)
        {
            Index_AddItems(index.Value, input.SelectMany(a => a).ToArray(), ids.ToArray(), length, 1);
        }

        public IEnumerable<(int candidateIndex, float Score)> Search(float[] query, int maxItemsPerSeed)
        {        
            int qSize = 1;
            
            long[] rSizes = new long[qSize];
            ItemAndScore[] results2 = new ItemAndScore[qSize * maxItemsPerSeed];

            Index_Search1(index.Value, query, qSize, maxItemsPerSeed, results2, rSizes, 1);
            
            return results2.Select(a => ((int)a.Item, a.Score));
        }

        public void Dispose()
        {
            if (index != null)
            {
                Index_Delete(index.Value);
                index = null;
            }
        }
    }

    public static class TestCpp
    {
        internal const string path = $@"..\..\..\..\x64\Debug";

        [DllImport($@"{path}\HNSWDll.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int init();

        [DllImport($@"{path}\HNSWDll.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int dispose();
    }

    internal class ScoreAndSortHNSWCpp : ScoreAndSortBase, IDisposable
    {
        private int _countPerIteration = 10_000;

        public ScoreAndSortHNSWCpp(int maxDegreeOfParallelism, int maxScoredItems, string datasetName, float[][] embeddedVectorsList, Func<float[], float[], float> distanceFunction)
            : base(maxDegreeOfParallelism, maxScoredItems, datasetName, embeddedVectorsList, distanceFunction)
        {
        }
        
        public void Init(string modelPath, int maxElements, int mParam, int efConstruction)
        {
            BuildFileNames(modelPath, $"{DatasetName}-dim{Dimensionality}-m{mParam}-ef{efConstruction}", (EmbeddedVectorsList.Length, EmbeddedVectorsList[0].Length), out var graphFilename);

            var r1 = TestCpp.init();
            var r2 = TestCpp.dispose();

            if (!File.Exists(graphFilename))
            {
                using (var index = new Index("cosine", Dimensionality))
                {
                    index.Init(maxElements, mParam, efConstruction, 0);

                    int numberOfElements = EmbeddedVectorsList.Length;
                    int numberOfIterations = (int)Math.Ceiling(1f * numberOfElements / _countPerIteration);
                    int handled = 0;

                    for (int iterationIndex = 0; iterationIndex < numberOfIterations; iterationIndex++)
                    {
                        var take = Math.Min(_countPerIteration, numberOfElements - handled);
                        var ids = Enumerable.Range(handled, handled + take).Select(a => (long)a);
                        index.AddItems(EmbeddedVectorsList.Skip(handled).Take(take), ids, take);
                        handled += take;
                    }

                    index.Save(graphFilename);
                }
            }

            AnnIndex = new Index("cosine", Dimensionality); 
            AnnIndex.Load(graphFilename, maxElements);
        }

        protected override IEnumerable<(int candidateIndex, float Score)> CalculateScoresPerSeed(int seedIndex)
        {
            var res = AnnIndex.Search(EmbeddedVectorsList[seedIndex], MaxScoredItems);
            return res.Select(a => (a.candidateIndex, 1f - a.Score));
        }

        public void Dispose()
        {
            AnnIndex.Dispose();
        }

        public Index AnnIndex { get; set; }
        
        private static void BuildFileNames(string modelPath, string prefixName, (int vcount, int vsize) shape, out string graphFilename)
        {
            string GraphPathSuffix = "bin_gf";

            var outputFileNamePrefix = Path.Join(modelPath, $@"{prefixName}-{shape.vcount:D}-{shape.vsize:D}");
            graphFilename = $"{outputFileNamePrefix}.{GraphPathSuffix}";
        }
    }
}
