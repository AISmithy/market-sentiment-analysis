"""\
© 2025 Nishant Kumar. Confidential and Proprietary.
Unauthorized copying, distribution, modification, or use of this software
is strictly prohibited without express written permission.
"""

from django.core.management.base import BaseCommand
from django.core.cache import cache
from django.conf import settings

try:
    # django-redis provides get_redis_connection
    from django_redis import get_redis_connection
except Exception:
    get_redis_connection = None


class Command(BaseCommand):
    help = 'Clear history cache keys (history:* and internal index)'

    def add_arguments(self, parser):
        parser.add_argument('--pattern', type=str, default='history:*', help='Key pattern to delete')

    def handle(self, *args, **options):
        pattern = options.get('pattern')
        backend = settings.CACHES.get('default', {}).get('BACKEND', '')
        self.stdout.write(f'Using cache backend: {backend}')

        if get_redis_connection and 'redis' in backend:
            # use redis scan and delete
            try:
                conn = get_redis_connection('default')
                cursor = '0'
                total = 0
                while True:
                    cursor, keys = conn.scan(cursor=cursor, match=pattern, count=1000)
                    if keys:
                        conn.delete(*keys)
                        total += len(keys)
                    if cursor == 0 or cursor == '0':
                        break
                # Also delete index key if present
                try:
                    conn.delete('history_cache_index')
                except Exception:
                    pass
                self.stdout.write(self.style.SUCCESS(f'Deleted {total} keys from Redis matching {pattern}'))
                return
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error clearing keys from redis: {e}'))

        # Fallback: use index stored under history_cache_index (locmem case)
        idx = cache.get('history_cache_index') or {}
        count = 0
        for k in list(idx.keys()):
            try:
                cache.delete(k)
                count += 1
            except Exception:
                pass
        try:
            cache.delete('history_cache_index')
        except Exception:
            pass
        self.stdout.write(self.style.SUCCESS(f'Deleted {count} keys via index'))
