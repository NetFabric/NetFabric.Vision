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

        readonly int _rows;
        readonly int _columns;
        readonly GradientOperator _gradientOperator;
        readonly int _anchorScanInterval;
        readonly Mat _smoothed;
        readonly Mat _gradientMap, _directionMap;
        readonly Mat _gradientX, _gradientY;
        readonly Mat _magnitudeMap;
        readonly Mat _absGradientX, _absGradientY;
        readonly double[] _cummulativeGradientDistribution = new double[256];

        public EdgeDrawing(int rows, int columns, GradientOperator gradientOperator, int anchorScanInterval)
        {
            _rows = rows;
            _columns = columns;
            _gradientOperator = gradientOperator;
            _anchorScanInterval = anchorScanInterval;
            _smoothed = new Mat(rows, columns, Depth.U8, 1);
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

            // compute the anchors
            var anchors = ExtractAnchors(anchorThreshold, _gradientMap, _directionMap, _rows, _columns, _anchorScanInterval);

            // connect the anchors by smart routing
            foreach(var anchor in anchors)
            {
                DrawEdge(anchor.Y, anchor.X, destination, _gradientMap, _directionMap, _rows, _columns);
            }
        }

        void ComputeGradient(Arr source, double gradientThreshold)
        {
            // gaussian filtering
            CV.Smooth(source, _smoothed, SmoothMethod.Gaussian, 5, 5, 1.0);

            // calculate gradients
            switch(_gradientOperator)
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

        internal static List<Point> ExtractAnchors(int threshold, Arr gradientMap, Arr directionMap, int rows, int columns, int scanInterval)
        {
            var anchors = new List<Point>();

            // iterate through the Rows
            for(int row = 1, rowEnd = rows - 1; row < rowEnd; row += scanInterval)
            {
                // iterate through the columns
                for(int col = 1, colEnd = columns - 1; col < colEnd; col += scanInterval)
                {
                    var g = gradientMap.GetReal(row, col);

                    if(directionMap.GetReal(row, col) == HorizontalValue)
                    {
                        // compare to horizontal neighbors
                        if(Math.Abs(g - gradientMap.GetReal(row - 1, col)) > threshold &&
                            Math.Abs(g - gradientMap.GetReal(row + 1, col)) > threshold)
                        {
                            anchors.Add(new Point(col, row));
                        }
                    }
                    else
                    {
                        // compare to vertical neighbors
                        if(Math.Abs(g - gradientMap.GetReal(row, col - 1)) > threshold &&
                            Math.Abs(g - gradientMap.GetReal(row, col + 1)) > threshold)
                        {
                            anchors.Add(new Point(col, row));
                        }
                    }
                }
            }

            return anchors;
        }

        internal static void DrawEdge(int row, int column, Arr edgeMap, Arr gradientMap, Arr directionMap, int rows, int columns)
        {
            var edgel = edgeMap.GetReal(row, column);
            if(edgel != 0)
                return;

            var direction = directionMap.GetReal(row, column);
            if(direction == HorizontalValue)
            { // go horizontal
                GoLeft(row, column, edgeMap, gradientMap, directionMap, rows, columns);
                GoRight(row, column, edgeMap, gradientMap, directionMap, rows, columns);
            }
            else
            { // go vertical
                GoUp(row, column, edgeMap, gradientMap, directionMap, rows, columns);
                GoDown(row, column, edgeMap, gradientMap, directionMap, rows, columns);
            }
        }

        static void GoLeft(int row, int column, Arr edgeMap, Arr gradientMap, Arr directionMap, int rows, int columns)
        {
            double upGradient, straightGradient, downGradient, direction, edgel;
            do
            {
                edgeMap.SetReal(row, column, 255);

                --column; // left
                if(column == 0)
                    return;

                upGradient = gradientMap.GetReal(row - 1, column);
                straightGradient = gradientMap.GetReal(row, column);
                downGradient = gradientMap.GetReal(row + 1, column);

                if(upGradient > straightGradient && upGradient > downGradient)
                {
                    --row; // up
                    if(row == 0)
                        return;
                }
                else if(downGradient > straightGradient && downGradient > upGradient)
                {
                    ++row; // down
                    if(row == rows - 1)
                        return;
                }
                else if(straightGradient == 0)
                {
                    return;
                }

                edgel = edgeMap.GetReal(row, column);
                if(edgel != 0)
                    return;

                direction = directionMap.GetReal(row, column);
            }
            while(direction == HorizontalValue);

            // go vertical
            GoUp(row, column, edgeMap, gradientMap, directionMap, rows, columns);
            GoDown(row, column, edgeMap, gradientMap, directionMap, rows, columns);
        }

        static void GoRight(int row, int column, Arr edgeMap, Arr gradientMap, Arr directionMap, int rows, int columns)
        {
            double upGradient, straightGradient, downGradient, direction, edgel;
            do
            {
                edgeMap.SetReal(row, column, 255);

                ++column; // right
                if(column == columns - 1)
                    return;

                upGradient = gradientMap.GetReal(row - 1, column);
                straightGradient = gradientMap.GetReal(row, column);
                downGradient = gradientMap.GetReal(row + 1, column);

                if(upGradient > straightGradient && upGradient > downGradient)
                {
                    --row; // up
                    if(row == 0)
                        return;
                }
                else if(downGradient > straightGradient && downGradient > upGradient)
                {
                    ++row; // down
                    if(row == rows - 1)
                        return;
                }
                else if(straightGradient == 0)
                {
                    return;
                }

                edgel = edgeMap.GetReal(row, column);
                if(edgel != 0)
                    return;

                direction = directionMap.GetReal(row, column);
            }
            while(direction == HorizontalValue);

            // go vertical
            GoUp(row, column, edgeMap, gradientMap, directionMap, rows, columns);
            GoDown(row, column, edgeMap, gradientMap, directionMap, rows, columns);
        }

        static void GoUp(int row, int column, Arr edgeMap, Arr gradientMap, Arr directionMap, int rows, int columns)
        {
            double leftGradient, straightGradient, rightGradient, direction, edgel;
            do
            {
                edgeMap.SetReal(row, column, 255);

                --row; // up
                if(row == 0)
                    return;

                leftGradient = gradientMap.GetReal(row, column - 1);
                straightGradient = gradientMap.GetReal(row, column);
                rightGradient = gradientMap.GetReal(row, column + 1);

                if(leftGradient > straightGradient && leftGradient > rightGradient)
                {
                    --column; // left
                    if(column == 0)
                        return;
                }
                else if(rightGradient > straightGradient && rightGradient > leftGradient)
                {
                    ++column; // right
                    if(column == columns - 1)
                        return;
                }
                else if(straightGradient == 0)
                {
                    return;
                }

                edgel = edgeMap.GetReal(row, column);
                if(edgel != 0)
                    return;

                direction = directionMap.GetReal(row, column);
            }
            while(direction == VerticalValue);

            // go horizontal
            GoLeft(row, column, edgeMap, gradientMap, directionMap, rows, columns);
            GoRight(row, column, edgeMap, gradientMap, directionMap, rows, columns);
        }

        static void GoDown(int row, int column, Arr edgeMap, Arr gradientMap, Arr directionMap, int rows, int columns)
        {
            double leftGradient, straightGradient, rightGradient, direction, edgel;
            do
            {
                edgeMap.SetReal(row, column, 255);

                ++row; // down
                if(row == rows - 1)
                    return;

                leftGradient = gradientMap.GetReal(row, column - 1);
                straightGradient = gradientMap.GetReal(row, column);
                rightGradient = gradientMap.GetReal(row, column + 1);

                if(leftGradient > straightGradient && leftGradient > rightGradient)
                {
                    --column; // left
                    if(column == 0)
                        return;
                }
                else if(rightGradient > straightGradient && rightGradient > leftGradient)
                {
                    ++column; // right
                    if(column == columns - 1)
                        return;
                }
                else if(straightGradient == 0)
                {
                    return;
                }

                edgel = edgeMap.GetReal(row, column);
                if(edgel != 0)
                    return;

                direction = directionMap.GetReal(row, column);
            }
            while(direction == VerticalValue);

            // go horizontal
            GoLeft(row, column, edgeMap, gradientMap, directionMap, rows, columns);
            GoRight(row, column, edgeMap, gradientMap, directionMap, rows, columns);
        }

    }
}
