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
        [InlineData("Resources/lena512color.tiff", GradientOperator.Prewitt, 24)]
        [InlineData("Resources/lena512color.tiff", GradientOperator.Sobel, 24)]
        [InlineData("Resources/lena512color.tiff", GradientOperator.Scharr, 24)]
        public void ComputeGradient(string sourceFileName, GradientOperator gradientOperator, int gradientThreshold)
        {
            // Arrange
            var source = LoadImage(sourceFileName);
            var rows = source.Size.Height;
            var columns = source.Size.Width;
            var edgeDrawing = new EdgeDrawing(rows, columns);

            // Act
            edgeDrawing.ComputeGradient(source, gradientOperator, gradientThreshold);

            // Assert
            var gradientFileName = Path.Combine("Output", "ComputeGradient", $"{Path.GetFileNameWithoutExtension(sourceFileName)}_{gradientOperator}_{gradientThreshold}_GradientMap.png");
            var directionsFileName = Path.Combine("Output", "ComputeGradient", $"{Path.GetFileNameWithoutExtension(sourceFileName)}_{gradientOperator}_{gradientThreshold}_DirectionsMap.png");
            SaveImage(edgeDrawing._gradientMap, gradientFileName);
            SaveImage(edgeDrawing._directionMap, directionsFileName);
        }

        [Theory]
        [InlineData("Resources/lena512color.tiff", GradientOperator.Prewitt, 24, 5, 8)]
        [InlineData("Resources/lena512color.tiff", GradientOperator.Sobel, 24, 5, 8)]
        [InlineData("Resources/lena512color.tiff", GradientOperator.Scharr, 24, 5, 8)]
        public void ExtractAnchors(string sourceFileName,
            GradientOperator gradientOperator, int gradientThreshold,
            int anchorScanInterval, int anchorThreshold)
        {
            // Arrange
            var source = LoadImage(sourceFileName);
            var rows = source.Size.Height;
            var columns = source.Size.Width;
            var edgeDrawing = new EdgeDrawing(rows, columns);
            edgeDrawing.ComputeGradient(source, gradientOperator, gradientThreshold);

            // Act
            var anchors = edgeDrawing.ExtractAnchors(anchorScanInterval, anchorThreshold);

            // Assert
            var anchorMap = new Mat(rows, columns, Depth.U8, 1);
            anchorMap.Set(Scalar.All(0));
            foreach(var anchor in anchors)
                anchorMap.SetReal(anchor.Y, anchor.X, 255);
            var resultFileName = Path.Combine("Output", "ExtractAnchors", $"{Path.GetFileNameWithoutExtension(sourceFileName)}_{gradientOperator}_{gradientThreshold}_{anchorScanInterval}_{anchorThreshold}.png");
            SaveImage(anchorMap, resultFileName);
        }

        [Theory]
        [InlineData(4, 8, "Resources/Topal_Gradient.png", "Resources/Topal_Direction.png", "Resources/Topal_Edges.png", "Topal_DrawEdge_Differences.png")]
        public void DrawEdge(
            int row, int column,
            string gradientFileName, string directionFileName, string expectedFileName, string differencesFileName)
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
        [InlineData("Resources/lena512color.tiff", GradientOperator.Prewitt, 24, 5, 8)]
        [InlineData("Resources/lena512color.tiff", GradientOperator.Sobel, 24, 5, 8)]
        [InlineData("Resources/lena512color.tiff", GradientOperator.Scharr, 24, 5, 8)]
        public void DrawEdges(
            string sourceFileName,
            GradientOperator gradientOperator, int gradientThreshold,
            int anchorScanInterval, int anchorThreshold)
        {
            // Arrange
            var source = LoadImage(sourceFileName);
            var rows = source.Size.Height;
            var columns = source.Size.Width;
            var edgeDrawing = new EdgeDrawing(rows, columns);
            var edgesMap = new Mat(rows, columns, Depth.U8, 1);
            edgesMap.Set(Scalar.All(0));

            // Act
            edgeDrawing.DrawEdges(source, edgesMap, gradientOperator, gradientThreshold, anchorScanInterval, anchorThreshold);

            // Assert
            var resultFileName = Path.Combine("Output", "DrawEdges", $"{Path.GetFileNameWithoutExtension(sourceFileName)}_{gradientOperator}_{gradientThreshold}_{anchorScanInterval}_{anchorThreshold}_EdgesMap.png");
            SaveImage(edgesMap, resultFileName);
        }

        Arr LoadImage(string fileName) =>
            CV.LoadImage(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, fileName), LoadImageFlags.Grayscale);

        void SaveImage(Arr source, string fileName)
        {
            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, fileName);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            CV.SaveImage(path, source);
        }
    }
}
