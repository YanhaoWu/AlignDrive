"""Merge Bench2Drive route shards and print the benchmark summary."""

import argparse
import json
import math
from pathlib import Path
import sys


def merge_route_json(folder_path, expected_routes=220):
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    if expected_routes < 1:
        raise ValueError('expected_routes must be positive')
    records = []
    route_ids = set()
    for path in sorted(folder.glob('*.json')):
        if path.name == 'merged.json':
            continue
        with path.open() as stream:
            payload = json.load(stream)
        checkpoint = payload.get('_checkpoint') if isinstance(payload, dict) else None
        if not isinstance(checkpoint, dict) or 'records' not in checkpoint:
            continue
        for record in checkpoint['records']:
            route_id = record['route_id']
            if route_id in route_ids:
                raise ValueError('Duplicate route {} in {}; use one evaluation run only'.format(route_id, path))
            score = float(record['scores']['score_composed'])
            if not math.isfinite(score):
                raise ValueError('Nonfinite driving score for route ' + str(route_id))
            route_ids.add(route_id)
            record.pop('index', None)
            records.append(record)
    if not records:
        raise ValueError('No route records found in ' + str(folder))
    if len(records) > expected_routes:
        raise ValueError('Found more records than expected; check the result directory')
    success_count = sum(
        record['status'] in ('Completed', 'Perfect') and not any(
            infractions for name, infractions in record['infractions'].items()
            if name != 'min_speed_infractions'
        ) for record in records
    )
    result = {
        '_checkpoint': {'records': sorted(records, key=lambda record: record['route_id'], reverse=True)},
        'driving score': sum(float(record['scores']['score_composed']) for record in records) / expected_routes,
        'success rate': success_count / expected_routes,
        'eval num': len(records),
    }
    (folder / 'merged.json').write_text(json.dumps(result, indent=4) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', nargs='?', help='The evaluation result/ directory containing route JSON shards')
    parser.add_argument('-f', '--folder', help='Alternative to the positional result directory')
    parser.add_argument('--expected-routes', type=int, default=220)
    args = parser.parse_args(argv)
    if bool(args.result_dir) == bool(args.folder):
        parser.error('Provide exactly one result directory, either positional or with --folder')
    folder = args.result_dir or args.folder
    try:
        result = merge_route_json(folder, args.expected_routes)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, 'Merge failed: {}\n'.format(error))
    if result['eval num'] != args.expected_routes:
        print('INCOMPLETE: missing routes count as zero; this is not a complete benchmark result.', file=sys.stderr)
    print('Routes: {} / {}'.format(result['eval num'], args.expected_routes))
    print('Driving Score: {:.2f}'.format(result['driving score']))
    print('Success Rate: {:.2f}%'.format(100 * result['success rate']))
    print('Saved: {}'.format(Path(folder).expanduser().resolve() / 'merged.json'))


if __name__ == '__main__':
    main()
