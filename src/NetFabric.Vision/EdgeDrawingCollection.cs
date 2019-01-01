using System;
using OpenCV.Net;

namespace NetFabric.Vision
{
    public partial class EdgeDrawing
    {
        public Edges DrawEdges(Arr source, GradientOperator gradientOperator, int gradientThreshold, int anchorScanInterval, int anchorThreshold, int minLength = 0)
        {
            // compute the gradient and direction maps
            _gradients.ComputeGradient(source, gradientOperator, gradientThreshold);

            // compute the anchors
            var anchors = _gradients.ExtractAnchors(anchorScanInterval, anchorThreshold);

            // initialize the edges map
            if(_edgesMap is null)
                _edgesMap = new Mat(_rows, _columns, Depth.U8, 1);
            _edgesMap.Set(Scalar.All(0));

            // connect the anchors by smart routing
            var edges = new Edges();
            foreach(var anchor in anchors)
            {
                DrawEdge(anchor.Y, anchor.X, _gradients, _edgesMap, edges, _rows, _columns, minLength);
            }
            return edges;
        }

        public Edges DrawEdgesParameterFree(Arr source, GradientOperator gradientOperator, int gradientThreshold, int anchorScanInterval, int minLength = 0)
        {
            // compute the gradient and direction maps
            _gradients.ComputeGradient(source, gradientOperator, gradientThreshold);

            // compute the normalized cummulative histogram
            _gradients.CummulativeHistogram();

            // compute the anchors
            var anchors = _gradients.ExtractAnchors(anchorScanInterval);

            // connect the anchors by smart routing
            var edges = new Edges();
            foreach(var anchor in anchors)
            {
                DrawEdge(anchor.Y, anchor.X, _gradients, _edgesMap, edges, _rows, _columns, minLength);
            }
            return edges;
        }

        internal static void DrawEdge(int row, int column, Gradients gradients, Arr edgeMap, Edges edges, int rows, int columns, int minLength)
        {
            var edgel = edgeMap.GetReal(row, column);
            if(edgel != 0)
                return;

            Edge edge;
            var direction = gradients.DirectionMap.GetReal(row, column);
            if(direction == Gradients.HorizontalValue)
                edge = GoHorizontal(row, column, gradients, edgeMap, edges, rows, columns, minLength);
            else
                edge = GoVertical(row, column, gradients, edgeMap, edges, rows, columns, minLength);

            if(edge.Count >= minLength)
                edges.Add(edge);
        }

        static Edge GoHorizontal(int row, int column, Gradients gradients, Arr edgeMap, Edges edges, int rows, int columns, int minLength)
        {
            var left = GoLeft(row, column);
            var right = GoRight(row, column);

            if(left.Count >= minLength)
            {
                if(right.Count >= minLength)
                {
                    left.AddLastFrom(right);
                }
                return left;
            }
            else if(right.Count >= minLength)
            {
                return right;
            }
            else
            {
                return new Edge();
            }

            Edge GoLeft(int r, int c)
            {
                var edge = new Edge();
                double upGradient, straightGradient, downGradient, direction, edgel;
                do
                {
                    edgeMap.SetReal(r, c, 255);
                    edge.AddLast(new Point(c, r));

                    --c; // left
                    if(c == 0)
                        return edge;

                    upGradient = gradients.GradientMap.GetReal(r - 1, c);
                    straightGradient = gradients.GradientMap.GetReal(r, c);
                    downGradient = gradients.GradientMap.GetReal(r + 1, c);

                    if(upGradient > straightGradient && upGradient > downGradient)
                    {
                        --r; // up
                        if(r == 0)
                            return edge;
                    }
                    else if(downGradient > straightGradient && downGradient > upGradient)
                    {
                        ++r; // down
                        if(r == r - 1)
                            return edge;
                    }
                    else if(straightGradient == 0)
                    {
                        return edge;
                    }

                    edgel = edgeMap.GetReal(r, c);
                    if(edgel != 0)
                        return edge;

                    direction = gradients.DirectionMap.GetReal(r, c);
                }
                while(direction == Gradients.HorizontalValue);

                // go vertical
                var vertical = GoVertical(r, c, gradients, edgeMap, edges, rows, columns, minLength);
                if(!vertical.IsEmpty)
                {
                    var verticalHead = vertical.First.Value;
                    if(verticalHead.X == c && verticalHead.Y == r)
                    {
                        edge.AddLastFrom(vertical);
                    }
                    else
                    {
                        var verticalTail = vertical.Last.Value;
                        if(verticalTail.X == c && verticalTail.Y == r)
                        {
                            edge.AddLastFrom(vertical, true);
                        }
                        else
                        {
                            edges.Add(vertical);
                        }
                    }
                }

                if(edge.Count >= minLength)
                    return edge;

                return new Edge();
            }

            Edge GoRight(int r, int c)
            {
                var edge = new Edge();
                double upGradient, straightGradient, downGradient, direction, edgel;
                do
                {
                    edgeMap.SetReal(r, c, 255);

                    ++c; // right
                    if(c == columns - 1)
                        return edge;

                    upGradient = gradients.GradientMap.GetReal(r - 1, c);
                    straightGradient = gradients.GradientMap.GetReal(r, c);
                    downGradient = gradients.GradientMap.GetReal(r + 1, c);

                    if(upGradient > straightGradient && upGradient > downGradient)
                    {
                        --r; // up
                        if(r == 0)
                            return edge;
                    }
                    else if(downGradient > straightGradient && downGradient > upGradient)
                    {
                        ++r; // down
                        if(r == rows - 1)
                            return edge;
                    }
                    else if(straightGradient == 0)
                    {
                        return edge;
                    }

                    edgel = edgeMap.GetReal(r, c);
                    if(edgel != 0)
                        return edge;

                    direction = gradients.DirectionMap.GetReal(r, c);
                }
                while(direction == Gradients.HorizontalValue);

                // go vertical
                var vertical = GoVertical(r, c, gradients, edgeMap, edges, rows, columns, minLength);
                if(!vertical.IsEmpty)
                {
                    var verticalHead = vertical.First.Value;
                    if(verticalHead.X == c && verticalHead.Y == r)
                    {
                        edge.AddLastFrom(vertical);
                    }
                    else
                    {
                        var verticalTail = vertical.Last.Value;
                        if(verticalTail.X == c && verticalTail.Y == r)
                        {
                            edge.AddLastFrom(vertical, true);
                        }
                        else
                        {
                            edges.Add(vertical);
                        }
                    }
                }

                if(edge.Count >= minLength)
                    return edge;

                return new Edge();
            }
        }

        static Edge GoVertical(int row, int column, Gradients gradients, Arr edgeMap, Edges edges, int rows, int columns, int minLength)
        {
            var up = GoUp(row, column);
            var down = GoDown(row, column);

            if(up.Count >= minLength)
            {
                if(down.Count >= minLength)
                {
                    up.AddLastFrom(down);
                }
                return up;
            }
            else if(down.Count >= minLength)
            {
                return down;
            }
            else
            {
                return new Edge();
            }

            Edge GoUp(int r, int c)
            {
                var edge = new Edge();
                double leftGradient, straightGradient, rightGradient, direction, edgel;
                do
                {
                    edgeMap.SetReal(r, c, 255);

                    --r; // up
                    if(r == 0)
                        return edge;

                    leftGradient = gradients.GradientMap.GetReal(r, c - 1);
                    straightGradient = gradients.GradientMap.GetReal(r, c);
                    rightGradient = gradients.GradientMap.GetReal(r, c + 1);

                    if(leftGradient > straightGradient && leftGradient > rightGradient)
                    {
                        --c; // left
                        if(c == 0)
                            return edge;
                    }
                    else if(rightGradient > straightGradient && rightGradient > leftGradient)
                    {
                        ++c; // right
                        if(c == columns - 1)
                            return edge;
                    }
                    else if(straightGradient == 0)
                    {
                        return edge;
                    }

                    edgel = edgeMap.GetReal(r, c);
                    if(edgel != 0)
                        return edge;

                    direction = gradients.DirectionMap.GetReal(r, c);
                }
                while(direction == Gradients.VerticalValue);

                // go horizontal
                var horizontal = GoHorizontal(r, c, gradients, edgeMap, edges, rows, columns, minLength);
                if(!horizontal.IsEmpty)
                {
                    var horizontalHead = horizontal.First.Value;
                    if(horizontalHead.X == c && horizontalHead.Y == r)
                    {
                        edge.AddLastFrom(horizontal);
                    }
                    else
                    {
                        var horizontalTail = horizontal.Last.Value;
                        if(horizontalTail.X == c && horizontalTail.Y == r)
                        {
                            edge.AddLastFrom(horizontal, true);
                        }
                        else
                        {
                            edges.Add(horizontal);
                        }
                    }
                }

                if(edge.Count >= minLength)
                    return edge;

                return new Edge();
            }

            Edge GoDown(int r, int c)
            {
                var edge = new Edge();
                double leftGradient, straightGradient, rightGradient, direction, edgel;
                do
                {
                    edgeMap.SetReal(r, c, 255);

                    ++r; // down
                    if(r == rows - 1)
                        return edge;

                    leftGradient = gradients.GradientMap.GetReal(r, c - 1);
                    straightGradient = gradients.GradientMap.GetReal(r, c);
                    rightGradient = gradients.GradientMap.GetReal(r, c + 1);

                    if(leftGradient > straightGradient && leftGradient > rightGradient)
                    {
                        --c; // left
                        if(c == 0)
                            return edge;
                    }
                    else if(rightGradient > straightGradient && rightGradient > leftGradient)
                    {
                        ++c; // right
                        if(c == columns - 1)
                            return edge;
                    }
                    else if(straightGradient == 0)
                    {
                        return edge;
                    }

                    edgel = edgeMap.GetReal(r, c);
                    if(edgel != 0)
                        return edge;

                    direction = gradients.DirectionMap.GetReal(r, c);
                }
                while(direction == Gradients.VerticalValue);

                // go horizontal
                var horizontal = GoHorizontal(r, c, gradients, edgeMap, edges, rows, columns, minLength);
                if(!horizontal.IsEmpty)
                {
                    var horizontalHead = horizontal.First.Value;
                    if(horizontalHead.X == c && horizontalHead.Y == r)
                    {
                        edge.AddLastFrom(horizontal);
                    }
                    else
                    {
                        var horizontalTail = horizontal.Last.Value;
                        if(horizontalTail.X == c && horizontalTail.Y == r)
                        {
                            edge.AddLastFrom(horizontal, true);
                        }
                        else
                        {
                            edges.Add(horizontal);
                        }
                    }
                }

                if(edge.Count >= minLength)
                    return edge;

                return new Edge();
            }
        }
    }
}
