namespace HNSW
{
    public interface IProgressReporter
    {
        void Progress(int current, int total);
    }
}
