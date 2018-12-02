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
        readonly Mat _edgesMap;
        readonly double[] _cummulativeGradientDistribution = new double[256];

        public EdgeDrawing(int rows, int columns, GradientOperator gradientOperator, int anchorScanInterval, double smoothSigma, int minEdgePoints, int maxRecursionLevel)
        {
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
            _edgesMap = new Mat(rows, columns, Depth.U8, 1);
            _edgesMap.Set(Scalar.All(0));
        }

        public List<DoubleLinkedList<Point>> DrawEdges(Arr source, int anchorThreshold, int gradientThreshold)
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
            var anchorsMap = new Mat(source.Size.Width, source.Size.Height, Depth.U8, 1);
            anchorsMap.Set(Scalar.All(0));
            var anchorColor = new Scalar(255);
            foreach(var anchor in anchors)
                CV.Line(anchorsMap, anchor, anchor, anchorColor);

            SaveImage(anchorsMap, "ExtractAnchors.bmp");
#endif

            // connect the anchors by smart routing
            var edges = new List<DoubleLinkedList<Point>>();
            foreach(var anchor in anchors)
            {
                HandleAnchor(anchor, edges);
            }

#if DEBUG
            var edgesMap = new Mat(source.Size.Width, source.Size.Height, Depth.U8, 3);
            edgesMap.Set(Scalar.All(0));
            var edgeColor = new Scalar(255);
            Point previousPoint;
            foreach(var edge in edges)
            {
                previousPoint = edge.First.Value;
                foreach(var point in edge.EnumerateForward())
                {
                    CV.Line(edgesMap, previousPoint, point, edgeColor);
                    previousPoint = point;
                }
            }

            SaveImage(edgesMap, "EdgesMap.bmp");
#endif

            return edges;
        }

#if DEBUG
        void SaveImage(Mat source, string fileName) =>
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
                    var g = (byte)_gradientMap.GetReal(row, col);

                    if((byte)_directionMap.GetReal(row, col) == HorizontalValue)
                    {
                        // compare to horizontal neighbors
                        if(Math.Abs(g - (byte)_gradientMap.GetReal(row - 1, col)) > threshold &&
                            Math.Abs(g - (byte)_gradientMap.GetReal(row + 1, col)) > threshold)
                        {
                            anchors.Add(new Point(col, row));
                        }
                    }
                    else
                    {
                        // compare to vertical neighbors
                        if(Math.Abs(g - (byte)_gradientMap.GetReal(row, col - 1)) > threshold &&
                            Math.Abs(g - (byte)_gradientMap.GetReal(row, col + 1)) > threshold)
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
                    var g = (byte)_gradientMap.GetReal(row, col);

                    if((byte)_directionMap.GetReal(row, col) == HorizontalValue)
                    {
                        // compare to horizontal neighbors
                        if(g > (byte)_gradientMap.GetReal(row - 1, col) && g > (byte)_gradientMap.GetReal(row + 1, col))
                        {
                            anchors[g].Add(new Point(col, row));
                        }
                    }
                    else
                    {
                        // compare to vertical neighbors
                        if(g > (byte)_gradientMap.GetReal(row, col - 1) && g > (byte)_gradientMap.GetReal(row, col + 1))
                        {
                            anchors[g].Add(new Point(col, row));
                        }
                    }
                }
            }

            return anchors;
        }

        void HandleAnchor(Point anchor, List<DoubleLinkedList<Point>> edges)
        {
            var direction = (byte)_directionMap.GetReal(anchor.Y, anchor.X);
            if(direction == HorizontalValue)
            { // is horizontal
                var edge = GoHorizontal(anchor.Y, anchor.X, edges, 0);
                if(edge.Count >= _minEdgePoints)
                    edges.Add(edge);
            }
            else
            { // is vertical
                var edge = GoVertical(anchor.Y, anchor.X, edges, 0);
                if(edge.Count >= _minEdgePoints)
                    edges.Add(edge);
            }
        }

        DoubleLinkedList<Point> GoHorizontal(int row, int column, List<DoubleLinkedList<Point>> edges, int recursionLevel)
        {
            var edge = new DoubleLinkedList<Point>();

            if(recursionLevel < _maxRecursionLevel)
            {

                var left = GoLeft(row, column, edges, recursionLevel);
                if(left.Count >= _minEdgePoints)
                {
                    edge = DoubleLinkedList.AppendInPlace(edge, left, false, true);
                }

                // add the current point to back of edge
                edge.AddLast(new Point(column, row));

                var right = GoRight(row, column, edges, recursionLevel);
                if(right.Count >= _minEdgePoints)
                {
                    edge = DoubleLinkedList.AppendInPlace(edge, right, false, false);
                }

            }

            return edge;
        }

        DoubleLinkedList<Point> GoVertical(int row, int column, List<DoubleLinkedList<Point>> edges, int recursionLevel)
        {
            var edge = new DoubleLinkedList<Point>();

            if(recursionLevel < _maxRecursionLevel)
            {

                var down = GoDown(row, column, edges, recursionLevel);
                if(down.Count >= _minEdgePoints)
                {
                    edge = DoubleLinkedList.AppendInPlace(edge, down, false, true);
                }

                // add the current point to back of edge
                edge.AddLast(new Point(column, row));

                var up = GoUp(row, column, edges, recursionLevel);
                if(up.Count >= _minEdgePoints)
                {
                    edge = DoubleLinkedList.AppendInPlace(edge, up, false, false);
                }

            }

            return edge;
        }

        DoubleLinkedList<Point> GoLeft(int row, int column, List<DoubleLinkedList<Point>> edges, int recursionLevel)
        {
            var edge = new DoubleLinkedList<Point>();

            byte gradient, upGradient, straightGradient, downGradient;
            var direction = HorizontalValue;
            while(column > 0 && column < _gradientMap.Cols - 1 && row > 0 && row < _gradientMap.Rows - 1)
            {
                --column; // left

                upGradient = (byte)_gradientMap.GetReal(row - 1, column);
                straightGradient = (byte)_gradientMap.GetReal(row, column);
                downGradient = (byte)_gradientMap.GetReal(row + 1, column);

                if(upGradient > straightGradient && upGradient > downGradient)
                {
                    --row; // up
                    gradient = upGradient;
                }
                else if(downGradient > straightGradient && downGradient > upGradient)
                {
                    ++row; // down
                    gradient = downGradient;
                }
                else
                {
                    // straight
                    gradient = straightGradient;
                }

                // check if not an edgel
                if(gradient == 0)
                    break;

                // check if edgel already handled
                var edgel = _edgesMap.GetReal(row, column);
                if(edgel == EdgeValue)
                    break;

                // check if direction changed
                direction = (byte)_directionMap.GetReal(row, column);
                if(direction == HorizontalValue)
                {
                    // keeps this edgel
                    edge.AddLast(new Point(column, row));
                    edgel = EdgeValue;
                }
                else
                {
                    var adjacentEdge = GoVertical(row, column, edges, recursionLevel + 1);
                    HandleAdjacentEdge(row, column, edge, adjacentEdge, edges);

                    // break loop
                    break;
                }
            }

            return edge;
        }

        DoubleLinkedList<Point> GoRight(int row, int column, List<DoubleLinkedList<Point>> edges, int recursionLevel)
        {
            var edge = new DoubleLinkedList<Point>();

            byte gradient, upGradient, straightGradient, downGradient;
            var direction = HorizontalValue;
            while(column > 0 && column < _gradientMap.Cols - 1 && row > 0 && row < _gradientMap.Rows - 1)
            {
                ++column; // right

                upGradient = (byte)_gradientMap.GetReal(row - 1, column);
                straightGradient = (byte)_gradientMap.GetReal(row, column);
                downGradient = (byte)_gradientMap.GetReal(row + 1, column);

                if(upGradient > straightGradient && upGradient > downGradient)
                {
                    --row; // up
                    gradient = upGradient;
                }
                else if(downGradient > straightGradient && downGradient > upGradient)
                {
                    ++row; // down
                    gradient = downGradient;
                }
                else
                {
                    // straight
                    gradient = straightGradient;
                }

                // check if not an edgel
                if(gradient == 0)
                    break;

                // check if edgel already handled
                var edgel = _edgesMap.GetReal(row, column);
                if(edgel == EdgeValue)
                    break;

                // check if direction changed
                direction = (byte)_directionMap.GetReal(row, column);
                if(direction == HorizontalValue)
                {
                    // keeps this edgel
                    edge.AddLast(new Point(column, row));
                    edgel = EdgeValue;
                }
                else
                {
                    var adjacentEdge = GoVertical(row, column, edges, recursionLevel + 1);
                    HandleAdjacentEdge(row, column, edge, adjacentEdge, edges);

                    // break loop
                    break;
                }
            }

            return edge;

        }

        DoubleLinkedList<Point> GoUp(int row, int column, List<DoubleLinkedList<Point>> edges, int recursionLevel)
        {
            var edge = new DoubleLinkedList<Point>();

            byte gradient, leftGradient, straightGradient, rightGradient;
            var direction = VerticalValue;
            while(column > 0 && column < _gradientMap.Cols - 1 && row > 0 && row < _gradientMap.Rows - 1)
            {
                --row; // up

                leftGradient = (byte)_gradientMap.GetReal(row, column - 1);
                straightGradient = (byte)_gradientMap.GetReal(row, column);
                rightGradient = (byte)_gradientMap.GetReal(row, column + 1);

                if(leftGradient > straightGradient && leftGradient > rightGradient)
                {
                    --column; // left
                    gradient = leftGradient;
                }
                else if(rightGradient > straightGradient && rightGradient > leftGradient)
                {
                    ++column; // right
                    gradient = rightGradient;
                }
                else
                {
                    // straight
                    gradient = straightGradient;
                }

                // check if not an edgel
                if(gradient == 0)
                    break;

                // check if edgel already handled
                var edgel = _edgesMap.GetReal(row, column);
                if(edgel == EdgeValue)
                    break;

                // check if direction changed
                direction = (byte)_directionMap.GetReal(row, column);
                if(direction == VerticalValue)
                {
                    // keeps this edgel
                    edge.AddLast(new Point(column, row));
                    edgel = EdgeValue;
                }
                else
                {
                    var adjacentEdge = GoHorizontal(row, column, edges, recursionLevel + 1);
                    HandleAdjacentEdge(row, column, edge, adjacentEdge, edges);

                    // break loop
                    break;
                }
            }

            return edge;

        }

        DoubleLinkedList<Point> GoDown(int row, int column, List<DoubleLinkedList<Point>> edges, int recursionLevel)
        {
            var edge = new DoubleLinkedList<Point>();

            byte gradient, leftGradient, straightGradient, rightGradient;
            byte direction = VerticalValue;
            while(column > 0 && column < _gradientMap.Cols - 1 && row > 0 && row < _gradientMap.Rows - 1)
            {
                ++row; // down

                leftGradient = (byte)_gradientMap.GetReal(row, column - 1);
                straightGradient = (byte)_gradientMap.GetReal(row, column);
                rightGradient = (byte)_gradientMap.GetReal(row, column + 1);

                if(leftGradient > straightGradient && leftGradient > rightGradient)
                {
                    --column; // left
                    gradient = leftGradient;
                }
                else if(rightGradient > straightGradient && rightGradient > leftGradient)
                {
                    ++column; // right
                    gradient = rightGradient;
                }
                else
                {
                    // straight
                    gradient = straightGradient;
                }

                // check if not an edgel
                if(gradient == 0)
                    break;

                // check if edgel already handled
                var edgel = _edgesMap.GetReal(row, column);
                if(edgel == EdgeValue)
                    break;

                // check if direction changed
                direction = (byte)_directionMap.GetReal(row, column);
                if(direction == VerticalValue)
                {
                    // keeps this edgel
                    edge.AddLast(new Point(column, row));
                    edgel = EdgeValue;
                }
                else
                {
                    var adjacentEdge = GoHorizontal(row, column, edges, recursionLevel + 1);
                    HandleAdjacentEdge(row, column, edge, adjacentEdge, edges);

                    // break loop
                    break;
                }
            }

            return edge;
        }

        void HandleAdjacentEdge(int row, int column, DoubleLinkedList<Point> edge, DoubleLinkedList<Point> adjacentEdge, List<DoubleLinkedList<Point>> edges)
        {
            if(adjacentEdge.Count == 0)
                return; // do nothing

            // append edges if they share extremities
            var adjacentFirst = adjacentEdge.First.Value;
            if(column == adjacentFirst.X && row == adjacentFirst.Y)
            {
                edge = DoubleLinkedList.AppendInPlace(edge, adjacentEdge, false, false);
                return;
            }

            var adjacentLast = adjacentEdge.Last.Value;
            if(column == adjacentLast.X && row == adjacentLast.Y)
            {
                edge = DoubleLinkedList.AppendInPlace(edge, adjacentEdge, false, true);
                return;
            }

            // edges don't share extremities
            // store adjacentEdge in edges
            edges.Add(adjacentEdge);
        }

    }
}
