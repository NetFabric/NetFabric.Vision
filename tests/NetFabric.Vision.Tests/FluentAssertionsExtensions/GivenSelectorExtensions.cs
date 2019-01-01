using System;
using FluentAssertions.Execution;
using OpenCV.Net;

namespace NetFabric.Vision.Tests
{
    static class GivenSelectorExtensions
    {
        public static ContinuationOfGiven<Arr> AssertArrIsNotNull(this GivenSelector<Arr> givenSelector) =>
            givenSelector
                .ForCondition(items => !(items is null))
                .FailWith("but found arr is <null>.");

        public static ContinuationOfGiven<Arr> AssertEitherArrHaveSameSizeAndElementType(this GivenSelector<Arr> givenSelector, Size size, int elementType) =>
            givenSelector
                .ForCondition(items => items.Size != size)
                .FailWith("but found size {0}.", items => items.Size)
                .Then
                .ForCondition(items => items.ElementType != elementType)
                .FailWith("but found {0}.", items => items.ElementType);

        public static void AssertArrsHaveSameValues(this GivenSelector<Arr> givenSelector, Arr expected) =>
            givenSelector
                .Given(actual => GetFirstDifferenceWith(actual, expected))
                .ForCondition(diff => diff.HasValue)
                .FailWith("but differs at position {0}.", diff => diff.Value);

        static Point? GetFirstDifferenceWith(Arr first, Arr second)
        {
            var difference = new Mat(first.Size, Depth.U8, 1);
            CV.Cmp(first, second, difference, ComparisonOperation.NotEqual);
            for(int row = 1, rowEnd = difference.Rows - 1; row < rowEnd; row++)
            {
                for(int col = 1, colEnd = difference.Cols - 1; col < colEnd; col++)
                {
                    if(difference.GetReal(row, col) != 0.0)
                    {
                        return new Point(col, row);
                    }
                }
            }
            return null;
        }
    }
}
