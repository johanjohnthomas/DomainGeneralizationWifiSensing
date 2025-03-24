import json
import argparse
from datetime import datetime

def generate_metadata(args):
    return {
        "experiment": {
            "name": args.model,
            "type": args.type,
            "research_question": args.rq,
            "date": datetime.now().isoformat()
        },
        "data": {
            "training_domains": "AR7a" if args.type == "leave_one_out" else "Multiple",
            "test_domains": "AR6a",
            "bandwidth": 80
        },
        "paths": {
            "metrics": f"{args.path.replace('meta.json','')}metrics/",
            "model": f"{args.path.replace('meta.json','')}model/",
            "plots": f"{args.path.replace('meta.json','')}plots/"
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--type", required=True)
    parser.add_argument("--rq", required=True)
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    
    metadata = generate_metadata(args)
    with open(args.path, 'w') as f:
        json.dump(metadata, f, indent=2)
