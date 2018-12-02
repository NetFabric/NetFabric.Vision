using System;
using System.IO;
using FluentAssertions;
using OpenCV.Net;
using Xunit;

namespace NetFabric.Vision.Tests
{
    public class DrawEdgesTests
    {
        [Fact]
        public void DrawEdges()
        {
            // Arrange
            var inputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "lena512color.tiff");
            var image = CV.LoadImage(inputPath, LoadImageFlags.Grayscale);

            // Act
            // TODO

            // Assert
            var outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "DrawEdges.tiff");
            CV.SaveImage(outputPath, image);
        }
    }
}
