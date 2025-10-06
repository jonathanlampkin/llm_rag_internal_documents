import argparse
from .probe import probe_pdf
from .cluster import cluster_features
from .route import route_pages
from .runners.tesseract import run_pdf

def main():
    parser = argparse.ArgumentParser("ocr-router")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe")
    p_probe.add_argument("input")
    p_probe.add_argument("--out", required=True)

    p_clust = sub.add_parser("cluster")
    p_clust.add_argument("features")
    p_clust.add_argument("--out", required=True)
    p_clust.add_argument("--k", type=int, default=8)

    p_route = sub.add_parser("route")
    p_route.add_argument("clusters")
    p_route.add_argument("--features", default=None)
    p_route.add_argument("--out", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("routes")
    p_run.add_argument("--input", default=None)
    p_run.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.cmd == "probe":
        probe_pdf(args.input, args.out)
    elif args.cmd == "cluster":
        cluster_features(args.features, args.out, k=args.k)
    elif args.cmd == "route":
        features = args.features or args.clusters.replace("clusters","features")
        route_pages(features, args.clusters, args.out)
    elif args.cmd == "run":
        inp = args.input or args.routes.split(".routes.")[0] + ".pdf"
        run_pdf(inp, args.routes, args.out)

if __name__ == "__main__":
    main()

