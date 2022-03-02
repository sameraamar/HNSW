// ----------------------------------------------------------------------------
//  Copyrhs (c) Microsoft Corporation. All rhss reserved. 
// ----------------------------------------------------------------------------
// 	

namespace Microsoft.Xbox.Recommendations.Modeling.ETS.Instances.Common
{
    using System;
    using System.Numerics;
    using System.Runtime.CompilerServices;

    public static class VectorUtilities
    {
        private static readonly int _vs1 = Vector<float>.Count;
        private static readonly int _vs2 = 2 * Vector<float>.Count;
        private static readonly int _vs3 = 3 * Vector<float>.Count;
        private static readonly int _vs4 = 4 * Vector<float>.Count;

        #region DotProduct

        /// <summary>
        /// Calculate and return the dot product between two float vectors.
        /// Assumption: The length of two vectors is the same
        /// </summary>
        /// <param name="lhs">first vector</param>
        /// <param name="rhs">second vector</param>
        /// <returns>dot product between lhs and rhs</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotProductVectorsScore(ReadOnlySpan<float> lhs, ReadOnlySpan<float> rhs)
        {
            var score = 0F;
            for (var i = 0; i < lhs.Length; i++)
            {
                score += lhs[i] * rhs[i];
            }

            return score;
        }

        /// <summary>
        /// Use SIMD to calculate and return the dot product between two float vectors.
        /// Assumption: The length of two vectors is the same
        /// </summary>
        /// <param name="lhs">first vector (as span)</param>
        /// <param name="rhs">second vector (as span)</param>
        /// <returns>dot product between lhs and rhs</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SIMDDotProductVectorsScore(ReadOnlySpan<float> lhs, ReadOnlySpan<float> rhs)
        {
            var result = 0f;
            var count = lhs.Length;
            var offset = 0;

            while (count >= _vs4)
            {
                result += Vector.Dot(new Vector<float>(lhs.Slice(offset)), new Vector<float>(rhs.Slice(offset)));
                result += Vector.Dot(new Vector<float>(lhs.Slice(offset + _vs1)), new Vector<float>(rhs.Slice(offset + _vs1)));
                result += Vector.Dot(new Vector<float>(lhs.Slice(offset + _vs2)), new Vector<float>(rhs.Slice(offset + _vs2)));
                result += Vector.Dot(new Vector<float>(lhs.Slice(offset + _vs3)), new Vector<float>(rhs.Slice(offset + _vs3)));
                if (count == _vs4)
                    return result;
                count -= _vs4;
                offset += _vs4;
            }

            if (count >= _vs2)
            {
                result += Vector.Dot(new Vector<float>(lhs.Slice(offset)), new Vector<float>(rhs.Slice(offset)));
                result += Vector.Dot(new Vector<float>(lhs.Slice(offset + _vs1)), new Vector<float>(rhs.Slice(offset + _vs1)));
                if (count == _vs2)
                    return result;
                count -= _vs2;
                offset += _vs2;
            }

            if (count >= _vs1)
            {
                result += Vector.Dot(new Vector<float>(lhs.Slice(offset)), new Vector<float>(rhs.Slice(offset)));
                if (count == _vs1)
                    return result;
                count -= _vs1;
                offset += _vs1;
            }

            if (count > 0)
            {
                while (count > 0)
                {
                    result += lhs[offset] * rhs[offset];
                    offset++;
                    count--;
                }
            }

            return result;
        }

        #endregion

        #region CosineSimilairy

        /// <summary>
        /// Calculate and return cosine similarity between two vectors.
        /// Assumption: The length of two vectors is the same
        /// </summary>
        /// <param name="lhs">vector of float (as span)</param>
        /// <param name="rhs">vector of float (as span)</param>
        /// <returns>cosine similarity</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CosineSimilarityVectorsScore(ReadOnlySpan<float> lhs, ReadOnlySpan<float> rhs)
        {
            var score = 0F;
            var scoreNorm1 = 0F;
            var scoreNorm2 = 0F;

            for (var i = 0; i < lhs.Length; i++)
            {
                score += lhs[i] * rhs[i];
                scoreNorm1 += lhs[i] * lhs[i];
                scoreNorm2 += rhs[i] * rhs[i];
            }

            return (float) (score / (Math.Sqrt(scoreNorm1) * Math.Sqrt(scoreNorm2)));
        }

        /// <summary>
        /// Calculates cosine similarity optimized using SIMD instructions.
        /// Assumption: The length of two vectors is the same
        /// </summary>
        /// <param name="u">lhs vector.</param>
        /// <param name="v">rhs vector.</param>
        /// <returns>Cosine similarity between u and v.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SIMDCosineSimilarityVectorsScore(ReadOnlySpan<float> u, ReadOnlySpan<float> v)
        {
            var dot = 0F;
            var norm = default(Vector2);
            var step = Vector<float>.Count;
            int i, to = u.Length - step;

            for (i = 0; i <= to; i += step)
            {
                var ui = new Vector<float>(u.Slice(i));
                var vi = new Vector<float>(v.Slice(i));
                dot += Vector.Dot(ui, vi);
                norm.X += Vector.Dot(ui, ui);
                norm.Y += Vector.Dot(vi, vi);
            }

            for (; i < u.Length; ++i)
            {
                dot += u[i] * v[i];
                norm.X += u[i] * u[i];
                norm.Y += v[i] * v[i];
            }

            norm = Vector2.SquareRoot(norm);
            float n = (norm.X * norm.Y);

            if (n == 0)
            {
                return 0f;
            }

            return dot / n;
        }

        /// <summary>
        /// Calculates cosine distance with assumption that u and v are unit vectors using SIMD instructions.
        /// Assumption: The length of two vectors is the same
        /// </summary>
        /// <param name="u">lhs vector.</param>
        /// <param name="v">rhs vector.</param>
        /// <returns>Cosine distance between u and v.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SIMDCosineSimilarityVectorsScoreForUnits(ReadOnlySpan<float> lhs, ReadOnlySpan<float> rhs)
        {
            return SIMDDotProductVectorsScore(lhs, rhs);
        }

        #endregion

        #region Euclidean

        /// <summary>
        /// Calculates euclidean squared distance optimized using SIMD instructions.
        /// Assumption: The length of two vectors is the same
        /// </summary>
        /// <param name="u">lhs vector.</param>
        /// <param name="v">rhs vector.</param>
        /// <returns>euclidean squared distance between u and v.</returns>
        public static float SIMDEuclideanSquared(float[] u, float[] v)
        {
            var norm = 0F;
            var step = Vector<float>.Count;

            int i, to = u.Length - step;
            for (i = 0; i <= to; i += step)
            {
                var ui = new Vector<float>(u, i);
                var vi = new Vector<float>(v, i);

                var s = Vector.Subtract(ui, vi);
                norm += Vector.Dot(s, s);
            }

            return norm;
        }

        #endregion

        #region HelperMethod

        /// <summary>
        /// return true if hardware acceleration is enabled.
        /// </summary>
        /// <returns>true or false</returns>
        public static bool IsHardwareAccelerated()
        {
            return Vector.IsHardwareAccelerated;
        }

        /// <summary>
        /// A utility to compare between two double numbers up to an epsilon.
        /// </summary>
        /// <param name="x">first number</param>
        /// <param name="y">second number</param>
        /// <param name="epsilon">epsilon value</param>
        /// <returns>true or false</returns>
        public static bool CompareAlmostEqual(double x, double y, double epsilon = 1.0E-005f)
        {
            // it is not good enough to test Abs(a-b) < eps
            // Basically we better check if the diff is too small in comparison to b
            // So, it would be better to test Abs(a-b)/b < eps
            // The below code does this but also handles special cases when b is zero

            // This is based upon implementation:
            // http://floating-point-gui.de/errors/comparison/

            if (x == y)
                return true;

            var absX = Math.Abs(x);
            var absY = Math.Abs(y);
            var diff = Math.Abs(x - y);

            if (x * y == 0)
                return diff < (epsilon * epsilon);
            else if (absX + absY == diff) // [1]
                return diff < epsilon;
            else
                return diff / (absX + absY) < epsilon;
        }

        #endregion

        #region Magnitude and normalize

        /// <summary>
        /// Use SIMD acceleration in order to calculate vector magnitude
        /// </summary>
        /// <param name="vector"></param>
        /// <returns>return the magnitude of the input vector</returns>
        public static float SIMDMagnitude(float[] vector)
        {
            var magnitude = 0.0f;
            var step = Vector<float>.Count;

            int i, to = vector.Length - step;
            for (i = 0; i <= to; i += Vector<float>.Count)
            {
                var vi = new Vector<float>(vector, i);
                magnitude += Vector.Dot(vi, vi);
            }

            for (; i < vector.Length; ++i)
            {
                magnitude += vector[i] * vector[i];
            }

            return (float)Math.Sqrt(magnitude);
        }

        /// <summary>
        /// Use SIMD acceleration in order to normalize the given vector
        /// </summary>
        /// <param name="vector">vector of float</param>
        public static void SIMDNormalize(float[] vector)
        {
            var normFactor = 1f / SIMDMagnitude(vector);
            var step = Vector<float>.Count;

            int i, to = vector.Length - step;
            for (i = 0; i <= to; i += step)
            {
                var vi = new Vector<float>(vector, i);
                vi = Vector.Multiply(normFactor, vi);
                vi.CopyTo(vector, i);
            }

            for (; i < vector.Length; ++i)
            {
                vector[i] *= normFactor;
            }
        }
        
        #endregion
    }
}