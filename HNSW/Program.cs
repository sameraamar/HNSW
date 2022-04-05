#define RANDOM
//#define WINES

// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");

int? maxDataSize;
string inputPath;
string outputPath;
string inputReducedDataFileName;
string inputOriginalDataFileName;

string datasetName = "Random100k40d";
var enableCSharpHnsw = true;

if (datasetName == "Random1m")
{
    maxDataSize = 1_000_000;

    inputPath = @"C:\temp\hnsw\Random";
    outputPath = @"C:\temp\hnsw\Random\results";

    inputReducedDataFileName = @"1mcosine-128.npz";
    inputOriginalDataFileName = @"1mcosine-1024.csv";

    enableCSharpHnsw = false;
}

else if (datasetName == "Random10k")
{
    maxDataSize = 1_000_000;

    inputPath = @"C:\temp\hnsw\Random10k";
    outputPath = @"C:\temp\hnsw\Random10k\results";

    inputReducedDataFileName = @"10kcosine-128.npz";
    inputOriginalDataFileName = @"10kcosine-1024.npz";
}

else if (datasetName == "Random100k")
{
    maxDataSize = 100_000;

    inputPath = @"C:\temp\hnsw\Random100k";
    outputPath = @"C:\temp\hnsw\Random100k\results";

    inputReducedDataFileName = @"100kcosine-128.npz";
    inputOriginalDataFileName = @"100kcosine-1024.npz";
    enableCSharpHnsw = false;
}

else if (datasetName == "Random100k40d")
{
    maxDataSize = 100_000;

    inputPath = @"C:\temp\hnsw\Random100k40d";
    outputPath = @"C:\temp\hnsw\Random100k40d\results";


    inputReducedDataFileName = @"100k40dcosine-20.npz";
    inputOriginalDataFileName = @"100k40dcosine-40.npz";
    
    enableCSharpHnsw = false;
}

else if (datasetName == "Wines")
{
    maxDataSize = 100000;

    inputPath = @"C:\Users\saaamar\OneDrive - Microsoft\temp\hnsw-benchmark\datasets\wines";
    outputPath = @"c:/temp/hnsw/results";

    inputReducedDataFileName = @"wines120kcosine-128.npz";
    inputOriginalDataFileName = @"wines120kcosine-1024.npz";
}

else
{
    throw new Exception("bad dataset name");
}


try
{
    var gPriorityQueueSearch = new ExactSearchTester(datasetName, inputPath, inputReducedDataFileName, inputOriginalDataFileName, debugMode: false, useHeapSort: true);
    gPriorityQueueSearch.Run(maxDegreeOfParallelism: 1, maxDataSize);

    var mParam = 32;
    var efConstruction = 800;

    var cpp = new CppHnswTester(gPriorityQueueSearch, mParam, efConstruction, outputPath);
    cpp.Run(maxDegreeOfParallelism: 1, gPriorityQueueSearch.groundTruthResults, gPriorityQueueSearch.groundTruthRuntime);

    var rd = new ReducedDimensionHnswTester(gPriorityQueueSearch);
    cpp.Run(maxDegreeOfParallelism: 1, gPriorityQueueSearch.groundTruthResults, gPriorityQueueSearch.groundTruthRuntime);

    if (enableCSharpHnsw)
    {
        var csharp = new CSharpHnswTester(gPriorityQueueSearch, mParam, efConstruction, outputPath);
        csharp.Run(maxDegreeOfParallelism: 1, gPriorityQueueSearch.groundTruthResults, gPriorityQueueSearch.groundTruthRuntime);
    }

}
catch (Exception ex)
{
    Console.WriteLine(ex.ToString());
    throw;
}

Console.WriteLine("Done.");