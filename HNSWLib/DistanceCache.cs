﻿// <copyright file="DistanceCache.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>


namespace HNSW
{
    using System;
    using System.Linq;
    using System.Runtime.CompilerServices;

    internal class DistanceCache<TDistance> where TDistance : struct
    {
        private TDistance[][] values = null;
        private bool[][] isNull = null;
        
        internal int HitCount;

        internal DistanceCache()
        {
        }

        internal void Resize(int pointsCount, bool overwrite)
        {
            int start0 = 0;

            if (values is null || overwrite)
            {
                values = new TDistance[pointsCount][];
                isNull = new bool[pointsCount][];
            }
            else if (pointsCount < values.Length)
            {
                return;
            }
            else
            {
                start0 = values.Length;
                Array.Resize(ref values, (int)pointsCount);
                Array.Resize(ref isNull, (int)pointsCount);

            }
            
            for (int i = start0; i < pointsCount; i++)
            {
                values[i] = Enumerable.Repeat(default(TDistance), i + 1).ToArray();
                isNull[i] = Enumerable.Repeat(false, i + 1).ToArray();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal TDistance GetValue(int fromId, int toId, Func<int,int,TDistance> getter)
        {
            if (fromId < toId)
            {
                int t = fromId;
                fromId = toId;
                toId = t;
            }
            
            if (isNull[fromId][toId])
            {
                HitCount++;
                return values[fromId][toId];
            }

            var d = getter(fromId, toId);
            values[fromId][toId] = d;
            isNull[fromId][toId] = true;

            return d;
        }
    }

    internal class DistanceCache1<TDistance> where TDistance : struct
    {
        /// <summary>
        /// https://referencesource.microsoft.com/#mscorlib/system/array.cs,2d2b551eabe74985,references
        /// We use powers of 2 for efficient modulo
        /// 2^28 = 268435456
        /// 2^29 = 536870912
        /// 2^30 = 1073741824
        /// </summary>
        private const int MaxArrayLength = 268435456; // 0x10000000;

        private TDistance[] values;

        private long[] keys;

        internal int HitCount;

        internal DistanceCache1()
        {
        }

        internal void Resize(int pointsCount, bool overwrite)
        {
            if(pointsCount <=0) { pointsCount = 1024; }

            long capacity = ((long)pointsCount * (pointsCount + 1)) >> 1;
            capacity = capacity < MaxArrayLength ? capacity : MaxArrayLength;

            if (keys is null || capacity > keys.Length || overwrite)
            {
                int i0 = 0;
                if (keys is null || overwrite)
                {
                    keys   = new long[(int)capacity];
                    values = new TDistance[(int)capacity];
                }
                else
                {
                    i0 = keys.Length;
                    Array.Resize(ref keys,   (int)capacity);
                    Array.Resize(ref values, (int)capacity);
                }

                // TODO: may be there is a better way to warm up cache and force OS to allocate pages
                keys.AsSpan().Slice(i0).Fill(-1);
                values.AsSpan().Slice(i0).Fill(default);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal TDistance GetValue(int fromId, int toId, Func<int,int,TDistance> getter)
        {
            long key = MakeKey(fromId, toId);
            int hash = (int)(key & (keys.Length - 1));

            if (keys[hash] == key)
            {
                HitCount++;
                return values[hash];
            }
            else
            {
                var d = getter(fromId, toId);
                keys[hash] = key;
                values[hash] = d;
                return d;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SetValue(int fromId, int toId, TDistance distance)
        {
            long key = MakeKey(fromId, toId);
            int hash = (int)(key & (keys.Length - 1));
            keys[hash] = key;
            values[hash] = distance;
        }

        /// <summary>
        /// Builds key for the pair of points.
        /// MakeKey(fromId, toId) == MakeKey(toId, fromId)
        /// </summary>
        /// <param name="fromId">The from point identifier.</param>
        /// <param name="toId">The to point identifier.</param>
        /// <returns>Key of the pair.</returns>
        private static long MakeKey(int fromId, int toId)
        {
            return fromId > toId ? (((long)fromId * (fromId + 1)) >> 1) + toId : (((long)toId * (toId + 1)) >> 1) + fromId;
        }
    }
}
