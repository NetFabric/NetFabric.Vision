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
        [InlineData("lena512color.tiff", GradientOperator.Prewitt, 4, 1.0, 1, 255, 8, 30)]
        public void DrawEdges(
            string fileName, GradientOperator gradientOperator,
            int anchorScanInterval, double smoothSigma,
            int minEdgePoints, int maxRecursionLevel, int anchorThreshold, int gradientThreshold)
        {
            var inputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, fileName);
            var image = CV.LoadImage(inputPath, LoadImageFlags.Grayscale);
            var rows = image.Size.Height;
            var columns = image.Size.Width;

            var edgeDrawing = new EdgeDrawing(rows, columns, gradientOperator, anchorScanInterval, smoothSigma, minEdgePoints, maxRecursionLevel);

            var anchors = edgeDrawing.DrawEdges(image, anchorThreshold, gradientThreshold);

        }

    }
}
