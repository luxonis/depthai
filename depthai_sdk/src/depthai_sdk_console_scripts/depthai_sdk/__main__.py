import argparse

import depthai_sdk


def main():
    parser = argparse.ArgumentParser(description='DepthAI SDK command-line interface.')
    subparsers = parser.add_subparsers(dest='command')

    sentry_parser = subparsers.add_parser('sentry', help='Enable or disable Sentry reporting')
    sentry_parser.add_argument('action', choices=['enable', 'disable', 'status'], help='Action to perform')

    args = parser.parse_args()

    if args.command == 'sentry':
        if args.action == 'enable':
            depthai_sdk.set_sentry_status(True)
            print('Sentry reporting was enabled.')
        elif args.action == 'disable':
            depthai_sdk.set_sentry_status(False)
            print('Sentry reporting was disabled.')
        elif args.action == 'status':
            status = depthai_sdk.get_config_field("sentry")
            print(f'Sentry is {"enabled" if status else "disabled"}.')


if __name__ == '__main__':
    main()
