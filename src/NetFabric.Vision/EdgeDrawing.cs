using System;
using System.Collections.Generic;
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

        public List<DoubleLinkedList<Point>> DrawEdges(Mat src, int anchorThreshold, int gradientThreshold)
        {
            // gaussian filtering
            CV.Smooth(src, _smooth, SmoothMethod.Gaussian, 5, 5, _smoothSigma);

            // compute the gradient and direction maps
            Utils.ComputeGradient(_smooth,
                _gradientOperator, gradientThreshold,
                _gradientX, _gradientY,
                _absGradientX, _absGradientY,
                _gradientMap, _directionMap);

            // compute the anchors
            var anchors = ExtractAnchors(anchorThreshold);

            // connect the anchors by smart routing
            var edges = new List<DoubleLinkedList<Point>>();
            foreach(var anchor in anchors)
            {
                HandleAnchor(anchor, edges);
            }

            return edges;
        }

        unsafe List<Point> ExtractAnchors(int anchorThreshold)
        {
            var anchors = new List<Point>();

            // iterate through the Rows
            for(int row = 1, rowEnd = _gradientMap.Rows - 1; row < rowEnd; row += _anchorScanInterval)
            {
                // get pointers to the data Rows for efficient data access
                var previousGradientRowPtr = (byte*)_gradientMap.Ptr(row - 1).ToPointer();
                var currentGradientRowPtr = (byte*)_gradientMap.Ptr(row).ToPointer();
                var nextGradientRowPtr = (byte*)_gradientMap.Ptr(row + 1).ToPointer();
                var currentDirectionRowPtr = (byte*)_directionMap.Ptr(row).ToPointer();

                // iterate through the columns
                for(int col = 1, colEnd = _gradientMap.Cols - 1; col < colEnd; col += _anchorScanInterval)
                {

                    var g = currentGradientRowPtr[col];

                    if(currentDirectionRowPtr[col] == HorizontalValue)
                    {
                        // compare to horizontal neighbors
                        if(g - previousGradientRowPtr[col] > anchorThreshold &&
                            g - nextGradientRowPtr[col] > anchorThreshold)
                        {
                            anchors.Add(new Point(col, row));
                        }
                    } else
                    {
                        // compare to vertical neighbors
                        if(g - currentGradientRowPtr[col - 1] > anchorThreshold &&
                            g - currentGradientRowPtr[col + 1] > anchorThreshold)
                        {
                            anchors.Add(new Point(col, row));
                        }
                    }
                }
            }

            return anchors;
        }

        unsafe DoubleLinkedList<Point>[] ExtractAnchorsParameterFree()
        {
            var anchors = new DoubleLinkedList<Point>[256];

            // iterate through the Rows
            for(int row = 1, rowEnd = _gradientMap.Rows - 1; row < rowEnd; row += _anchorScanInterval)
            {

                var previousGradientRowPtr = (byte*)_gradientMap.Ptr(row - 1).ToPointer();
                var currentGradientRowPtr = (byte*)_gradientMap.Ptr(row).ToPointer();
                var nextGradientRowPtr = (byte*)_gradientMap.Ptr(row + 1).ToPointer();
                var currentDirectionRowPtr = (byte*)_directionMap.Ptr(row).ToPointer();

                // iterate through the columns
                for(int col = 1, colEnd = _gradientMap.Cols - 1; col < colEnd; col += _anchorScanInterval)
                {

                    var g = currentGradientRowPtr[col];

                    if(currentDirectionRowPtr[col] == HorizontalValue)
                    {
                        // compare to horizontal neighbors
                        if(g > previousGradientRowPtr[col] && g > nextGradientRowPtr[col])
                        {
                            anchors[g].AddLast(new Point(col, row));
                        }
                    } else
                    {
                        // compare to vertical neighbors
                        if(g > currentGradientRowPtr[col - 1] && g > currentGradientRowPtr[col + 1])
                        {
                            anchors[g].AddLast(new Point(col, row));
                        }
                    }
                }
            }

            return anchors;
        }

        DoubleLinkedList<Point> GoLeft(Point point, List<DoubleLinkedList<Point>> edges, int recursionLevel)
        {
            var edge = new DoubleLinkedList<Point>();

            byte gradient, upGradient, straightGradient, downGradient;
            var direction = HorizontalValue;
            while(point.X > 0 && point.X < _gradientMap.Cols - 1 && point.Y > 0 && point.Y < _gradientMap.Rows - 1)
            {
                --point.X; // left

                upGradient = (byte)_gradientMap.GetReal(point.Y - 1, point.X);
                straightGradient = (byte)_gradientMap.GetReal(point.X, point.Y);
                downGradient = (byte)_gradientMap.GetReal(point.Y + 1, point.X);

                if(upGradient > straightGradient && upGradient > downGradient)
                {
                    --point.Y; // up
                    gradient = upGradient;
                } else if(downGradient > straightGradient && downGradient > upGradient)
                {
                    ++point.Y; // down
                    gradient = downGradient;
                } else
                {
                    // straight
                    gradient = straightGradient;
                }

                // check if not an edgel
                if(gradient == 0)
                    break;

                // check if edgel already handled
                var edgel = _edgesMap.GetReal(point.X, point.Y);
                if(edgel == EdgeValue)
                    break;

                // check if direction changed
                direction = (byte)_directionMap.GetReal(point.X, point.Y);
                if(direction == HorizontalValue)
                {
                    // keeps this edgel
                    edge.AddLast(point);
                    edgel = EdgeValue;
                } else
                {
                    var adjacentEdge = GoVertical(point, edges, recursionLevel + 1);
                    HandleAdjacentEdge(point, edge, adjacentEdge, edges);

                    // break loop
                    break;
                }
            }

            return edge;
        }

        DoubleLinkedList<Point> GoRight(Point point, List<DoubleLinkedList<Point>> edges, int recursionLevel)
        {
            var edge = new DoubleLinkedList<Point>();

            byte gradient, upGradient, straightGradient, downGradient;
            var direction = HorizontalValue;
            while(point.X > 0 && point.X < _gradientMap.Cols - 1 && point.Y > 0 && point.Y < _gradientMap.Rows - 1)
            {
                ++point.X; // right

                upGradient = (byte)_gradientMap.GetReal(point.Y - 1, point.X);
                straightGradient = (byte)_gradientMap.GetReal(point.X, point.Y);
                downGradient = (byte)_gradientMap.GetReal(point.Y + 1, point.X);

                if(upGradient > straightGradient && upGradient > downGradient)
                {
                    --point.Y; // up
                    gradient = upGradient;
                } else if(downGradient > straightGradient && downGradient > upGradient)
                {
                    ++point.Y; // down
                    gradient = downGradient;
                } else
                {
                    // straight
                    gradient = straightGradient;
                }

                // check if not an edgel
                if(gradient == 0)
                    break;

                // check if edgel already handled
                var edgel = _edgesMap.GetReal(point.X, point.Y);
                if(edgel == EdgeValue)
                    break;

                // check if direction changed
                direction = (byte)_directionMap.GetReal(point.X, point.Y);
                if(direction == HorizontalValue)
                {
                    // keeps this edgel
                    edge.AddLast(point);
                    edgel = EdgeValue;
                } else
                {
                    var adjacentEdge = GoVertical(point, edges, recursionLevel + 1);
                    HandleAdjacentEdge(point, edge, adjacentEdge, edges);

                    // break loop
                    break;
                }
            }

            return edge;

        }

        DoubleLinkedList<Point> GoUp(Point point, List<DoubleLinkedList<Point>> edges, int recursionLevel)
        {
            var edge = new DoubleLinkedList<Point>();

            byte gradient, leftGradient, straightGradient, rightGradient;
            var direction = VerticalValue;
            while(point.X > 0 && point.X < _gradientMap.Cols - 1 && point.Y > 0 && point.Y < _gradientMap.Rows - 1)
            {
                --point.Y; // up

                leftGradient = (byte)_gradientMap.GetReal(point.Y, point.X - 1);
                straightGradient = (byte)_gradientMap.GetReal(point.X, point.Y);
                rightGradient = (byte)_gradientMap.GetReal(point.Y, point.X + 1);

                if(leftGradient > straightGradient && leftGradient > rightGradient)
                {
                    --point.X; // left
                    gradient = leftGradient;
                } else if(rightGradient > straightGradient && rightGradient > leftGradient)
                {
                    ++point.X; // right
                    gradient = rightGradient;
                } else
                {
                    // straight
                    gradient = straightGradient;
                }

                // check if not an edgel
                if(gradient == 0)
                    break;

                // check if edgel already handled
                var edgel = _edgesMap.GetReal(point.X, point.Y);
                if(edgel == EdgeValue)
                    break;

                // check if direction changed
                direction = (byte)_directionMap.GetReal(point.X, point.Y);
                if(direction == VerticalValue)
                {
                    // keeps this edgel
                    edge.AddLast(point);
                    edgel = EdgeValue;
                } else
                {
                    var adjacentEdge = GoHorizontal(point, edges, recursionLevel + 1);
                    HandleAdjacentEdge(point, edge, adjacentEdge, edges);

                    // break loop
                    break;
                }
            }

            return edge;

        }

        DoubleLinkedList<Point> GoDown(Point point, List<DoubleLinkedList<Point>> edges, int recursionLevel)
        {
            var edge = new DoubleLinkedList<Point>();

            byte gradient, leftGradient, straightGradient, rightGradient;
            byte direction = VerticalValue;
            while(point.X > 0 && point.X < _gradientMap.Cols - 1 && point.Y > 0 && point.Y < _gradientMap.Rows - 1)
            {
                ++point.Y; // down

                leftGradient = (byte)_gradientMap.GetReal(point.Y, point.X - 1);
                straightGradient = (byte)_gradientMap.GetReal(point.X, point.Y);
                rightGradient = (byte)_gradientMap.GetReal(point.Y, point.X + 1);

                if(leftGradient > straightGradient && leftGradient > rightGradient)
                {
                    --point.X; // left
                    gradient = leftGradient;
                } else if(rightGradient > straightGradient && rightGradient > leftGradient)
                {
                    ++point.X; // right
                    gradient = rightGradient;
                } else
                {
                    // straight
                    gradient = straightGradient;
                }

                // check if not an edgel
                if(gradient == 0)
                    break;

                // check if edgel already handled
                var edgel = _edgesMap.GetReal(point.X, point.Y);
                if(edgel == EdgeValue)
                    break;

                // check if direction changed
                direction = (byte)_directionMap.GetReal(point.X, point.Y);
                if(direction == VerticalValue)
                {
                    // keeps this edgel
                    edge.AddLast(point);
                    edgel = EdgeValue;
                } else
                {
                    var adjacentEdge = GoHorizontal(point, edges, recursionLevel + 1);
                    HandleAdjacentEdge(point, edge, adjacentEdge, edges);

                    // break loop
                    break;
                }
            }

            return edge;
        }

        void HandleAdjacentEdge(Point point, DoubleLinkedList<Point> edge, DoubleLinkedList<Point> adjacentEdge, List<DoubleLinkedList<Point>> edges)
        {
            if(adjacentEdge.Count == 0)
                return; // do nothing

            // append edges if they share extremities
            var adjacentFirst = adjacentEdge.First.Value;
            if(point.X == adjacentFirst.X && point.Y == adjacentFirst.Y)
            {
                edge = DoubleLinkedList.AppendInPlace(edge, adjacentEdge, false, false);
                return;
            }

            var adjacentLast = adjacentEdge.Last.Value;
            if(point.X == adjacentLast.X && point.Y == adjacentLast.Y)
            {
                edge = DoubleLinkedList.AppendInPlace(edge, adjacentEdge, false, true);
                return;
            }

            // edges don't share extremities
            // store adjacentEdge in edges
            edges.Add(adjacentEdge);
        }

        DoubleLinkedList<Point> GoHorizontal(Point point, List<DoubleLinkedList<Point>> edges, int recursionLevel)
        {
            var edge = new DoubleLinkedList<Point>();

            if(recursionLevel < _maxRecursionLevel)
            {

                var left = GoLeft(point, edges, recursionLevel);
                if(left.Count >= _minEdgePoints)
                {
                    edge = DoubleLinkedList.AppendInPlace(edge, left, false, true);
                }

                // add the current point to back of edge
                edge.AddLast(point);

                var right = GoRight(point, edges, recursionLevel);
                if(right.Count >= _minEdgePoints)
                {
                    edge = DoubleLinkedList.AppendInPlace(edge, right, false, false);
                }

            }

            return edge;
        }

        DoubleLinkedList<Point> GoVertical(Point point, List<DoubleLinkedList<Point>> edges, int recursionLevel)
        {
            var edge = new DoubleLinkedList<Point>();

            if(recursionLevel < _maxRecursionLevel)
            {

                var down = GoDown(point, edges, recursionLevel);
                if(down.Count >= _minEdgePoints)
                {
                    edge = DoubleLinkedList.AppendInPlace(edge, down, false, true);
                }

                // add the current point to back of edge
                edge.AddLast(point);

                var up = GoUp(point, edges, recursionLevel);
                if(up.Count >= _minEdgePoints)
                {
                    edge = DoubleLinkedList.AppendInPlace(edge, up, false, false);
                }

            }

            return edge;
        }

        void HandleAnchor(Point anchor, List<DoubleLinkedList<Point>> edges)
        {
            var direction = (byte)_directionMap.GetReal(anchor.Y, anchor.X);
            if(direction == HorizontalValue)
            { // is horizontal
                var edge = GoHorizontal(anchor, edges, 0);
                if(edge.Count >= _minEdgePoints)
                    edges.Add(edge);
            } else
            { // is vertical
                var edge = GoVertical(anchor, edges, 0);
                if(edge.Count >= _minEdgePoints)
                    edges.Add(edge);
            }
        }

    }
}
