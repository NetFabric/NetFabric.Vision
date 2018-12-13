using System;
using System.Collections.Generic;
using System.IO;
using OpenCV.Net;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("NetFabric.Vision.Tests")]

namespace NetFabric.Vision
{
    public enum GradientOperator
    {
        Prewitt,
        Sobel,
        Scharr,
    };

    public class EdgeDrawing
    {
        const byte HorizontalValue = 0;
        const byte VerticalValue = 255;
        const byte EdgeValue = 0;

        static readonly Mat PrewittX = Mat.FromArray(new float[] { -1, 0, 1, -1, 0, 1, -1, 0, 1 });
        static readonly Mat PrewittY = Mat.FromArray(new float[] { -1, -1, -1, 0, 0, 0, 1, 1, 1 });
        static readonly Mat ScharrX = Mat.FromArray(new float[] { -3, 0, 3, -10, 0, 10, -3, 0, 3 });
        static readonly Mat ScharrY = Mat.FromArray(new float[] { -3, -10, -3, 0, 0, 0, 3, 10, 3 });

        readonly int _rows;
        readonly int _columns;
        readonly GradientOperator _gradientOperator;
        readonly int _anchorScanInterval;
        readonly double _smoothSigma;
        readonly int _minEdgePoints;
        readonly int _maxRecursionLevel;
        readonly Mat _smooth;
        readonly Mat _gradientMap, _directionMap;
        readonly Mat _gradientX, _gradientY;
        readonly Mat _magnitudeMap;
        readonly Mat _absGradientX, _absGradientY;
        readonly double[] _cummulativeGradientDistribution = new double[256];

        public EdgeDrawing(int rows, int columns, GradientOperator gradientOperator, int anchorScanInterval, double smoothSigma, int minEdgePoints, int maxRecursionLevel)
        {
            _rows = rows;
            _columns = columns;
            _gradientOperator = gradientOperator;
            _anchorScanInterval = anchorScanInterval;
            _smoothSigma = smoothSigma;
            _minEdgePoints = minEdgePoints;
            _maxRecursionLevel = maxRecursionLevel;
            _smooth = new Mat(rows, columns, Depth.U8, 1);
            _gradientMap = new Mat(rows, columns, Depth.U8, 1);
            _directionMap = new Mat(rows, columns, Depth.U8, 1);
            _gradientX = new Mat(rows, columns, Depth.S16, 1);
            _gradientY = new Mat(rows, columns, Depth.S16, 1);
            _magnitudeMap = new Mat(rows, columns, Depth.U8, 1);
            _absGradientX = new Mat(rows, columns, Depth.U8, 1);
            _absGradientY = new Mat(rows, columns, Depth.U8, 1);
        }

        public void DrawEdges(Arr source, Arr destination, int anchorThreshold, int gradientThreshold)
        {
            // compute the gradient and direction maps
            ComputeGradient(source, gradientThreshold);

#if DEBUG
            SaveImage(_gradientMap, "GradientMap.bmp");
            SaveImage(_directionMap, "DirectionMap.bmp");
#endif

            // compute the anchors
            var anchors = ExtractAnchors(anchorThreshold);

#if DEBUG
            var anchorsMap = new Mat(_rows, _columns, Depth.U8, 1);
            anchorsMap.Set(Scalar.All(0));
            var anchorColor = new Scalar(255);
            foreach(var anchor in anchors)
                CV.Line(anchorsMap, anchor, anchor, anchorColor);

            SaveImage(anchorsMap, "ExtractAnchors.bmp");
#endif

            // connect the anchors by smart routing
            foreach(var anchor in anchors)
            {
                DrawEdges(anchor.Y, anchor.X, destination);
            }

#if DEBUG
            SaveImage(destination, "EdgesMap.bmp");
#endif
        }

#if DEBUG
        void SaveImage(Arr source, string fileName) =>
            CV.SaveImage(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, fileName), source);
#endif

        void ComputeGradient(Arr source, double gradientThreshold)
        {
            // gaussian filtering
            CV.Smooth(source, _smooth, SmoothMethod.Gaussian, 5, 5, _smoothSigma);

            // calculate gradients
            switch(_gradientOperator)
            {
                case GradientOperator.Prewitt:
                    CV.Filter2D(_smooth, _gradientX, PrewittX);
                    CV.Filter2D(_smooth, _gradientY, PrewittY);
                    if(gradientThreshold < 0.0)
                        gradientThreshold = 6.0;
                    break;
                case GradientOperator.Sobel:
                    CV.Sobel(_smooth, _gradientX, 1, 0);
                    CV.Sobel(_smooth, _gradientY, 0, 1);
                    break;
                case GradientOperator.Scharr:
                    CV.Filter2D(_smooth, _gradientX, PrewittX);
                    CV.Filter2D(_smooth, _gradientY, PrewittY);
                    break;
                default:
                    throw new Exception($"Unknown gradient operator: {_gradientOperator}");
            }

            // calculate absolute values for gradients
            CV.ConvertScaleAbs(_gradientX, _absGradientX);
            CV.ConvertScaleAbs(_gradientY, _absGradientY);

            // merge gradients  
            // d = 0.5 * abs(dx) + 0.5 * abs(dy)
            CV.AddWeighted(_absGradientX, 0.5, _absGradientY, 0.5, 0.0, _gradientMap);

            // eliminate gradient weak pixels
            CV.Threshold(_gradientMap, _gradientMap, gradientThreshold, 255, ThresholdTypes.ToZero);

            // edge direction 
            // abs(dx) >= abs(dy) => VERTICAL
            CV.Cmp(_absGradientX, _absGradientY, _directionMap, ComparisonOperation.GreaterOrEqual);
        }

        List<Point> ExtractAnchors(int threshold)
        {
            var anchors = new List<Point>();

            // iterate through the Rows
            for(int row = 1, rowEnd = _gradientMap.Rows - 1; row < rowEnd; row += _anchorScanInterval)
            {
                // iterate through the columns
                for(int col = 1, colEnd = _gradientMap.Cols - 1; col < colEnd; col += _anchorScanInterval)
                {
                    var g = _gradientMap.GetReal(row, col);

                    if(_directionMap.GetReal(row, col) == HorizontalValue)
                    {
                        // compare to horizontal neighbors
                        if(Math.Abs(g - _gradientMap.GetReal(row - 1, col)) > threshold &&
                            Math.Abs(g - _gradientMap.GetReal(row + 1, col)) > threshold)
                        {
                            anchors.Add(new Point(col, row));
                        }
                    }
                    else
                    {
                        // compare to vertical neighbors
                        if(Math.Abs(g - _gradientMap.GetReal(row, col - 1)) > threshold &&
                            Math.Abs(g - _gradientMap.GetReal(row, col + 1)) > threshold)
                        {
                            anchors.Add(new Point(col, row));
                        }
                    }
                }
            }

            return anchors;
        }

        List<Point>[] ExtractAnchorsParameterFree()
        {
            var anchors = new List<Point>[256];

            // iterate through the Rows
            for(int row = 1, rowEnd = _gradientMap.Rows - 1; row < rowEnd; row += _anchorScanInterval)
            {
                // iterate through the columns
                for(int col = 1, colEnd = _gradientMap.Cols - 1; col < colEnd; col += _anchorScanInterval)
                {
                    var g = (int)_gradientMap.GetReal(row, col);

                    if(_directionMap.GetReal(row, col) == HorizontalValue)
                    {
                        // compare to horizontal neighbors
                        if(g > _gradientMap.GetReal(row - 1, col) && g > _gradientMap.GetReal(row + 1, col))
                        {
                            anchors[g].Add(new Point(col, row));
                        }
                    }
                    else
                    {
                        // compare to vertical neighbors
                        if(g > _gradientMap.GetReal(row, col - 1) && g > _gradientMap.GetReal(row, col + 1))
                        {
                            anchors[g].Add(new Point(col, row));
                        }
                    }
                }
            }

            return anchors;
        }

        void DrawEdges(int row, int column, Arr edgeMap)
        {
            var edgel = edgeMap.GetReal(row, column);
            if(edgel != 0)
                return;

            var direction = _directionMap.GetReal(row, column);
            if(direction == HorizontalValue)
            { // go horizontal
                GoLeft(row, column, edgeMap);
                GoRight(row, column, edgeMap);
            }
            else
            { // go vertical
                GoUp(row, column, edgeMap);
                GoDown(row, column, edgeMap);
            }
        }

        void GoLeft(int row, int column, Arr edgeMap)
        {
            double upGradient, straightGradient, downGradient, direction, edgel;
            do
            {
                edgeMap.SetReal(row, column, 255);

                --column; // left
                if(column == 0)
                    return;

                upGradient = _gradientMap.GetReal(row - 1, column);
                straightGradient = _gradientMap.GetReal(row, column);
                downGradient = _gradientMap.GetReal(row + 1, column);

                if(upGradient > straightGradient && upGradient > downGradient)
                {
                    --row; // up
                    if(row == 0)
                        return;
                }
                else if(downGradient > straightGradient && downGradient > upGradient)
                {
                    ++row; // down
                    if(row == _rows - 1)
                        return;
                }
                else if(straightGradient == 0)
                {
                    return;
                }

                edgel = edgeMap.GetReal(row, column);
                if(edgel != 0)
                    return;

                direction = _directionMap.GetReal(row, column);
            }
            while(direction == HorizontalValue);

            // go vertical
            GoUp(row, column, edgeMap);
            GoDown(row, column, edgeMap);
        }

        void GoRight(int row, int column, Arr edgeMap)
        {
            double upGradient, straightGradient, downGradient, direction, edgel;
            do
            {
                edgeMap.SetReal(row, column, 255);

                ++column; // right
                if(column == _columns - 1)
                    return;

                upGradient = _gradientMap.GetReal(row - 1, column);
                straightGradient = _gradientMap.GetReal(row, column);
                downGradient = _gradientMap.GetReal(row + 1, column);

                if(upGradient > straightGradient && upGradient > downGradient)
                {
                    --row; // up
                    if(row == 0)
                        return;
                }
                else if(downGradient > straightGradient && downGradient > upGradient)
                {
                    ++row; // down
                    if(row == _rows - 1)
                        return;
                }
                else if(straightGradient == 0)
                {
                    return;
                }

                edgel = edgeMap.GetReal(row, column);
                if(edgel != 0)
                    return;

                direction = _directionMap.GetReal(row, column);
            }
            while(direction == HorizontalValue);

            // go vertical
            GoUp(row, column, edgeMap);
            GoDown(row, column, edgeMap);
        }

        void GoUp(int row, int column, Arr edgeMap)
        {
            double leftGradient, straightGradient, rightGradient, direction, edgel;
            do
            {
                edgeMap.SetReal(row, column, 255);

                --row; // up
                if(row == 0)
                    return;

                leftGradient = _gradientMap.GetReal(row, column - 1);
                straightGradient = _gradientMap.GetReal(row, column);
                rightGradient = _gradientMap.GetReal(row, column + 1);

                if(leftGradient > straightGradient && leftGradient > rightGradient)
                {
                    --column; // left
                    if(column == 0)
                        return;
                }
                else if(rightGradient > straightGradient && rightGradient > leftGradient)
                {
                    ++column; // right
                    if(column == _columns - 1)
                        return;
                }
                else if(straightGradient == 0)
                {
                    return;
                }

                edgel = edgeMap.GetReal(row, column);
                if(edgel != 0)
                    return;

                direction = _directionMap.GetReal(row, column);
            }
            while(direction == VerticalValue);

            // go horizontal
            GoLeft(row, column, edgeMap);
            GoRight(row, column, edgeMap);
        }

        void GoDown(int row, int column, Arr edgeMap)
        {
            double leftGradient, straightGradient, rightGradient, direction, edgel;
            do
            {
                edgeMap.SetReal(row, column, 255);

                ++row; // down
                if(row == _rows - 1)
                    return;

                leftGradient = _gradientMap.GetReal(row, column - 1);
                straightGradient = _gradientMap.GetReal(row, column);
                rightGradient = _gradientMap.GetReal(row, column + 1);

                if(leftGradient > straightGradient && leftGradient > rightGradient)
                {
                    --column; // left
                    if(column == 0)
                        return;
                }
                else if(rightGradient > straightGradient && rightGradient > leftGradient)
                {
                    ++column; // right
                    if(column == _columns - 1)
                        return;
                }
                else if(straightGradient == 0)
                {
                    return;
                }

                edgel = edgeMap.GetReal(row, column);
                if(edgel != 0)
                    return;

                direction = _directionMap.GetReal(row, column);
            }
            while(direction == VerticalValue);

            // go horizontal
            GoLeft(row, column, edgeMap);
            GoRight(row, column, edgeMap);
        }

    }
}
