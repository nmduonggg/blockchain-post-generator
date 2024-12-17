from datetime import datetime


def to_datetime(timestamp):
    dt_object = datetime.utcfromtimestamp(timestamp)
    return dt_object.strftime('%Y-%m-%d %H:%M').replace(' ', '\n')