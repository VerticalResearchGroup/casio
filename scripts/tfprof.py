import re
import utils

blacklist_terms = {
    'Memcpy',
    'Cast',
    'BroadcastGradientArgs'
}

def read_tf_prof_file(filename):
    with open(filename) as f:
        capture = False
        line = next(f)

        try:
            while True:
                while line.strip() != 'Profile:':
                    line = next(f)

                next(f)
                line = next(f)

                while 'End of Report' not in line:
                    if line.strip() != '':
                        ignore = False
                        for term in blacklist_terms:
                            if term in line:
                                ignore = True

                        if not ignore:
                            yield line

                    line = next(f)

        except StopIteration:
            pass

def parse_tf_op(line):
    if '@@void' in line:
        line = line.replace('@@void', '').strip()

    regex = r"([\w/]+)\s+[\w.]+\s\((\d+.\d+)%,\s*(\d+.\d+)%\),\s+([\w.]+)\s+\((\d+.\d+)%,\s*(\d+.\d+)%\),\s+([\w.]+)\s+\((\d+.\d+)%,\s*(\d+.\d+)%\),\s+([\w.]+)\s+\((\d+.\d+)%,\s*(\d+.\d+)%\)"
    m = re.match(regex, line)

    assert m, f'Failed to parse line: {line}'

    accel_time_str = m.group(7)

    if accel_time_str.endswith('ms'):
        accel_time = float(accel_time_str[:-2]) / 1e3
    elif accel_time_str.endswith('us'):
        accel_time = float(accel_time_str[:-2]) / 1e6
    else:
        assert False, f'Unknown time units: {accel_time_str}'

    return utils.FrameworkOp(
        name=m.group(1),
        accel_time=accel_time,
    )


def parse_tf_prof(filename):
    return [
        op for op in map(parse_tf_op, read_tf_prof_file(filename))
        if op.accel_time > 0
    ]

