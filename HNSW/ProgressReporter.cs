using System;

namespace HNSW
{
    using System.Diagnostics;
    using HNSW;

    internal class ProgressReporter : IProgressReporter
    {              
        internal ProgressReporter(Stopwatch clock)
        {
            Clock = clock;
        }
        
        public Stopwatch Clock { get; set; }

        public void Progress(int current, int total)
        {
            var m = total / 10;
            if (m>0 && current % m == 0)
            {
                //var hits = string.Join('\n', CosineDistanceExt.dictionHits.OrderByDescending(v => v.Value).Take(10).Select(a => "\t\t" + a));
                //var hits = string.Join('\n', CosineDistanceExt.dictionHits.OrderByDescending(v => v.Value).Take(10).Select(a => "\t\t" + a.Value));
                Console.WriteLine($"\tProgress: {current} / {total} [elapsed {Clock.ElapsedMilliseconds} ms] - distance()"); // called {counter} times");
            }

        }
    }
}
