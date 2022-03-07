// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");

var t = new HnswTester(false);
t.Run(maxDegreeOfParallelism:1);
Console.WriteLine("Done.");