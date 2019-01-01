using System;
using System.Collections.Generic;
using OpenCV.Net;

namespace NetFabric.Vision
{
    class Gradients
    {
        public const byte HorizontalValue = 0;
        public const byte VerticalValue = 255;

        static readonly Mat PrewittX = Mat.FromArray(new float[] { -1, 0, 1, -1, 0, 1, -1, 0, 1 });
        static readonly Mat PrewittY = Mat.FromArray(new float[] { -1, -1, -1, 0, 0, 0, 1, 1, 1 });

        readonly int _rows;
        readonly int _columns;
        readonly Mat _smoothed;
        readonly Mat _gradientX, _gradientY;
        readonly Mat _magnitudeMap;
        readonly Mat _absGradientX, _absGradientY;

        public Gradients(int rows, int columns)
        {
            _rows = rows;
            _columns = columns;
            _smoothed = new Mat(rows, columns, Depth.U8, 1);
            _gradientX = new Mat(rows, columns, Depth.S16, 1);
            _gradientY = new Mat(rows, columns, Depth.S16, 1);
            _magnitudeMap = new Mat(rows, columns, Depth.U8, 1);
            _absGradientX = new Mat(rows, columns, Depth.U8, 1);
            _absGradientY = new Mat(rows, columns, Depth.U8, 1);
            GradientMap = new Mat(rows, columns, Depth.U8, 1);
            DirectionMap = new Mat(rows, columns, Depth.U8, 1);
        }

        public Mat GradientMap { get; }

        public Mat DirectionMap { get; }

        public void ComputeGradient(Arr source, GradientOperator gradientOperator, double gradientThreshold)
        {
            // gaussian filtering
            CV.Smooth(source, _smoothed, SmoothMethod.Gaussian, 5, 5, 1.0);

            // calculate gradients
            switch(gradientOperator)
            {
                case GradientOperator.Prewitt:
                    CV.Filter2D(_smoothed, _gradientX, PrewittX);
                    CV.Filter2D(_smoothed, _gradientY, PrewittY);
                    if(gradientThreshold < 0.0)
                        gradientThreshold = 6.0;
                    break;
                case GradientOperator.Sobel:
                    CV.Sobel(_smoothed, _gradientX, 1, 0);
                    CV.Sobel(_smoothed, _gradientY, 0, 1);
                    break;
                case GradientOperator.Scharr:
                    CV.Sobel(_smoothed, _gradientX, 1, 0, -1);
                    CV.Sobel(_smoothed, _gradientY, 0, 1, -1);
                    break;
                default:
                    throw new Exception($"Unknown gradient operator: {gradientOperator}");
            }

            // calculate absolute values for gradients
            CV.ConvertScaleAbs(_gradientX, _absGradientX);
            CV.ConvertScaleAbs(_gradientY, _absGradientY);

            // merge gradients  
            // d = 0.5 * abs(dx) + 0.5 * abs(dy)
            CV.AddWeighted(_absGradientX, 0.5, _absGradientY, 0.5, 0.0, GradientMap);

            // eliminate gradient weak pixels
            CV.Threshold(GradientMap, GradientMap, gradientThreshold, 255, ThresholdTypes.ToZero);

            // edge direction 
            // abs(dx) >= abs(dy) => VERTICAL
            CV.Cmp(_absGradientX, _absGradientY, DirectionMap, ComparisonOperation.GreaterOrEqual);
        }

        internal List<Point> ExtractAnchors(int scanInterval, int threshold)
        {
            var anchors = new List<Point>();

            // iterate through the Rows
            for(int row = 1, rowEnd = _rows - 1; row < rowEnd; row += scanInterval)
            {
                // iterate through the columns
                for(int col = 1, colEnd = _columns - 1; col < colEnd; col += scanInterval)
                {
                    var gradient = GradientMap.GetReal(row, col);

                    if(DirectionMap.GetReal(row, col) == HorizontalValue)
                    {
                        // compare to horizontal neighbors
                        if(Math.Abs(gradient - GradientMap.GetReal(row - 1, col)) > threshold &&
                            Math.Abs(gradient - GradientMap.GetReal(row + 1, col)) > threshold)
                        {
                            anchors.Add(new Point(col, row));
                        }
                    }
                    else
                    {
                        // compare to vertical neighbors
                        if(Math.Abs(gradient - GradientMap.GetReal(row, col - 1)) > threshold &&
                            Math.Abs(gradient - GradientMap.GetReal(row, col + 1)) > threshold)
                        {
                            anchors.Add(new Point(col, row));
                        }
                    }
                }
            }

            return anchors;
        }

        public List<Point> ExtractAnchors(int scanInterval)
        {
            var anchors = new List<Point>();

            // iterate through the Rows
            for(int row = 1, rowEnd = _rows - 1; row < rowEnd; row += scanInterval)
            {
                // iterate through the columns
                for(int col = 1, colEnd = _columns - 1; col < colEnd; col += scanInterval)
                {
                    var gradient = GradientMap.GetReal(row, col);

                    if(DirectionMap.GetReal(row, col) == HorizontalValue)
                    {
                        // compare to horizontal neighbors
                        if(gradient > GradientMap.GetReal(row - 1, col) &&
                            gradient > GradientMap.GetReal(row + 1, col))
                        {
                            anchors.Add(new Point(col, row));
                        }
                    }
                    else
                    {
                        // compare to vertical neighbors
                        if(gradient > GradientMap.GetReal(row, col - 1) &&
                            gradient > GradientMap.GetReal(row, col + 1))
                        {
                            anchors.Add(new Point(col, row));
                        }
                    }
                }
            }

            return anchors;
        }

        public Histogram CummulativeHistogram()
        {
            var histogram = new Histogram(256, new[] { 0, 256 }, HistogramType.Array);
            histogram.CalcArrHist(new[] { GradientMap });
            histogram.Normalize(1.0);

            var cummulative = histogram.Bins.GetReal(0);
            for(var index = 1; index < 256; index++)
            {
                cummulative += histogram.Bins.GetReal(index);
                histogram.Bins.SetReal(index, cummulative);
            }

            return histogram;
        }

    }
}
