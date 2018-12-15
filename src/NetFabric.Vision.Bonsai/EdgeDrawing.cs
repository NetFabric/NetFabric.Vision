using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.Linq;
using System.Reactive.Linq;
using Bonsai;
using OpenCV.Net;

namespace NetFabric.Vision.Bonsai
{
    [Description("Edge segment detection.")]
    public class EdgeDrawing : Transform<IplImage, IplImage>
    {
        [Range(0, 6)]
        [Description("The order of the horizontal derivative.")]
        [Editor(DesignTypes.NumericUpDownEditor, typeof(UITypeEditor))]
        public int AnchorScanInterval { get; set; }

        public override IObservable<IplImage> Process(IObservable<IplImage> source)
        {
            return source.Select(input =>
            {
                return input;
            });
        }
    }
}

