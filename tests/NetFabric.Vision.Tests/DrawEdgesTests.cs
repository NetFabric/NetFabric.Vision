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
        [InlineData("lena512color.tiff", GradientOperator.Prewitt, 30.0, 1.0)]
        public void ComputeGradient(string fileName, GradientOperator gradientOperator, double gradientThreshold, double smoothSigma)
        {
            // Arrange
            var inputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, fileName);
            var image = CV.LoadImage(inputPath, LoadImageFlags.Grayscale);
            var rows = image.Size.Height;
            var columns = image.Size.Width;

            var smooth = new Mat(rows, columns, Depth.U8, 1);
            var gradientMap = new Mat(rows, columns, Depth.U8, 1);
            var directionMap = new Mat(rows, columns, Depth.U8, 1);
            var gradientX = new Mat(rows, columns, Depth.S16, 1);
            var gradientY = new Mat(rows, columns, Depth.S16, 1);
            var magnitudeMap = new Mat(rows, columns, Depth.U8, 1);
            var absGradientX = new Mat(rows, columns, Depth.U8, 1);
            var absGradientY = new Mat(rows, columns, Depth.U8, 1);

            CV.Smooth(image, smooth, SmoothMethod.Gaussian, 5, 5, smoothSigma);

            // Act
            Utils.ComputeGradient(smooth,
                gradientOperator, gradientThreshold,
                gradientX, gradientY,
                absGradientX, absGradientY,
                gradientMap, directionMap);

            // Assert
            SaveImage(gradientMap,
                $"ComputeGradient_GradientMap_{Path.GetFileNameWithoutExtension(fileName)}_{gradientOperator}_{gradientThreshold}_{smoothSigma}.bmp");
            SaveImage(directionMap,
                $"ComputeGradient_DirectionMap_{Path.GetFileNameWithoutExtension(fileName)}_{gradientOperator}_{gradientThreshold}_{smoothSigma}.bmp");
        }

        void SaveImage(Mat source, string fileName) =>
            CV.SaveImage(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, fileName), source);
    }
}
