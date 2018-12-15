using System;
using System.IO;
using FluentAssertions;
using OpenCV.Net;
using Xunit;

namespace NetFabric.Vision.Tests
{
    public class DrawEdgesTests
    {
        [Theory]
        [InlineData(4, 8, "Resources/Topal_Gradient.png", "Resources/Topal_Direction.png", "Resources/Topal_Edges.png", "Topal_DrawEdge_Differences.png")]
        public void DrawEdge(int row, int column, string gradientFileName, string directionFileName, string expectedFileName, string differencesFileName)
        {
            // Arrange
            var gradientMap = LoadImage(gradientFileName);
            var directionMap = LoadImage(directionFileName);
            var expectedMap = LoadImage(expectedFileName);
            var rows = gradientMap.Size.Height;
            var columns = gradientMap.Size.Width;
            var edgesMap = new Mat(rows, columns, Depth.U8, 1);
            edgesMap.Set(Scalar.All(0));
            var differencesMap = new Mat(rows, columns, Depth.U8, 1);

            // Act
            EdgeDrawing.DrawEdge(row, column, edgesMap, gradientMap, directionMap, rows, columns);

            // Assert
            CV.Cmp(edgesMap, expectedMap, differencesMap, ComparisonOperation.NotEqual);
            SaveImage(differencesMap, differencesFileName);
        }

        [Theory]
        [InlineData("Resources/lena512color.tiff", GradientOperator.Prewitt, 4, 1.0, 1, 255, 8, 30, "lena512color_DrawEdges_Edges.png")]
        public void DrawEdges(
                    string sourceFileName, GradientOperator gradientOperator,
                    int anchorScanInterval, double smoothSigma,
                    int minEdgePoints, int maxRecursionLevel, int anchorThreshold, int gradientThreshold, string resultFileName)
        {
            // Arrange
            var source = LoadImage(sourceFileName);
            var rows = source.Size.Height;
            var columns = source.Size.Width;
            var edgeDrawing = new EdgeDrawing(rows, columns, gradientOperator, anchorScanInterval, smoothSigma, minEdgePoints, maxRecursionLevel);
            var edgesMap = new Mat(rows, columns, Depth.U8, 1);
            edgesMap.Set(Scalar.All(0));

            // Act
            edgeDrawing.DrawEdges(source, edgesMap, anchorThreshold, gradientThreshold);

            // Assert
            SaveImage(edgesMap, resultFileName);
        }

        Arr LoadImage(string fileName) =>
            CV.LoadImage(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, fileName), LoadImageFlags.Grayscale);

        void SaveImage(Arr source, string fileName) =>
            CV.SaveImage(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, fileName), source);
    }
}
