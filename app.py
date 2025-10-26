#!/usr/bin/env python
"""
Convenience launcher for the Django development server.

Usage:
    python app.py             # starts the dev server (equivalent to `manage.py runserver`)
    python app.py 0.0.0.0:8000 # pass host:port or other manage.py args through

This is a small wrapper so contributors can run the app with a single file.
"""
import os
import sys


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'market_site.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and available on your PYTHONPATH?"
        ) from exc

    # If no args supplied, default to runserver
    if len(sys.argv) == 1:
        sys.argv += ['runserver']

    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
